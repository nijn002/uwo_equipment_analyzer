import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import os
import base64
import io

@dataclass
class EquipmentRequirement:
    required_skills: Dict[str, int]  # skill_name: minimum_level
    required_ex_skills: Set[str]
    min_attack: int = 0
    min_defense: int = 0
    gender: str = "All"

@dataclass
class Equipment:
    part: str
    name: str
    attack: int
    defense: int
    skills: Dict[str, int]
    ex_skill: Tuple[str, int]
    gender: str
    released: str = "Yes"

class EquipmentDatabase:
    def __init__(self):
        """Initialize an empty equipment database."""
        self.equipment_by_part = defaultdict(list)
        self.all_skills = set()
        
    def load_from_csv(self, df: pd.DataFrame) -> None:
        """Load equipment data from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing equipment data
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = {'Part', 'Name', 'Att', 'Def', 'Male/Female/All', 
                          'Skill_1', 'Boost_1', 'Skill_2', 'Boost_2', 'Skill_3', 'Boost_3',
                          'EX_Skill', 'Boost_Ex', 'Released'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        for _, row in df.iterrows():
            try:
                # Parse skills and boosts
                skills = {}
                skill_columns = {
                    1: ('Skill_1', 'Boost_1'),
                    2: ('Skill_2', 'Boost_2'),
                    3: ('Skill_3', 'Boost_3')
                }
                
                for i, (skill_col, boost_col) in skill_columns.items():
                    skill_name = str(row[skill_col]).strip() if pd.notna(row[skill_col]) else ''
                    boost_value = row[boost_col] if pd.notna(row[boost_col]) else '0'
                    if skill_name:
                        try:
                            boost = int(float(boost_value))
                            skills[skill_name] = boost
                            self.all_skills.add(skill_name)
                        except (ValueError, AttributeError) as e:
                            st.warning(f"Invalid boost value for {skill_name}: {boost_value}")
                            continue
                
                # Parse EX skill
                ex_skill = None
                if pd.notna(row['EX_Skill']):
                    try:
                        ex_boost = int(float(row['Boost_Ex'])) if pd.notna(row['Boost_Ex']) else 0
                        ex_skill = (str(row['EX_Skill']).strip(), ex_boost)
                        self.all_skills.add(str(row['EX_Skill']))
                    except (ValueError, AttributeError) as e:
                        st.warning(f"Invalid EX skill boost: {row['Boost_Ex']}")
                        pass
                
                # Parse released status
                released = "Yes"  # Default
                if pd.notna(row['Released']):
                    released = str(row['Released']).strip()
                
                # Validate gender
                gender = str(row['Male/Female/All']).strip()
                if gender not in ['Male', 'Female', 'All']:
                    st.warning(f"Invalid gender value: {gender}, defaulting to 'All'")
                    gender = 'All'
                
                equipment = Equipment(
                    part=str(row['Part']).strip(),
                    name=str(row['Name']).strip(),
                    attack=int(float(row['Att'])) if pd.notna(row['Att']) else 0,
                    defense=int(float(row['Def'])) if pd.notna(row['Def']) else 0,
                    skills=skills,
                    ex_skill=ex_skill,
                    gender=gender,
                    released=released
                )
                self.equipment_by_part[equipment.part].append(equipment)
            except Exception as e:
                st.error(f"Error processing row {_+1}: {str(e)}")
                st.error(f"Row data: {row.to_dict()}")
                continue

def format_equipment_name(equipment: Equipment) -> str:
    """Format equipment name with skills for display.
    
    Args:
        equipment (Equipment): The equipment to format, or None
        
    Returns:
        str: Formatted string representation of the equipment
    """
    if equipment is None:
        return "None"
    
    parts = [f"{equipment.name}     "]
    
    skill_parts = []
    for skill, boost in sorted(equipment.skills.items()):  # Sort skills alphabetically
        skill_parts.append(f"{skill}+{boost}")
    if skill_parts:
        parts.append("(" + ", ".join(skill_parts) + ")")
    
    if equipment.ex_skill:
        skill_name, boost = equipment.ex_skill
        parts.append(f" (EX: {skill_name} R{boost})")
    
    return "".join(parts)

# Add authentication
def check_password() -> bool:
    """Check if the user has entered the correct password.
    
    Returns:
        bool: True if password is correct, False otherwise
    """
    
    def password_entered() -> None:
        """Handle password entry and validation."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Please enter the password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Please enter the password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="GVO Equipment Analyzer",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚔️ GVO Equipment Analyzer")
    
    if not check_password():
        st.stop()  # Do not continue if password is incorrect
        return

    # Initialize session state
    if 'database' not in st.session_state:
        with st.spinner("Loading equipment database..."):
            st.session_state.database = EquipmentDatabase()
            try:
                # Try to get CSV data from secrets
                csv_content = st.secrets["equipment_csv"]
                # Convert base64 string back to CSV data
                csv_data = base64.b64decode(csv_content).decode('utf-8')
                df = pd.read_csv(io.StringIO(csv_data))
                st.session_state.database.load_from_csv(df)
                st.session_state.df = df
                st.success("Equipment database loaded successfully!")
            except Exception as e:
                st.error("Error loading equipment data. Please contact the administrator.")
                st.error(f"Error details: {str(e)}")
                st.stop()
                return

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Equipment Selection")
        
        # Equipment selection for each part
        for part in ['Weapon', 'Head', 'Body', 'Hands', 'Feet', 'Acc']:
            with st.container():
                equipment_list = st.session_state.database.equipment_by_part[part]
                
                # Create equipment options, sorted alphabetically by name
                options = ["None"] + sorted([format_equipment_name(e) for e in equipment_list])
                
                # Add search box for filtering equipment
                search_term = st.text_input(
                    f"🔍 Search {part}",
                    key=f"search_{part}",
                    help=f"Search {part} equipment by name or skills"
                )
                
                # Filter options based on search term
                if search_term:
                    search_term = search_term.lower()
                    filtered_options = ["None"] + [opt for opt in options[1:] 
                                                if search_term in opt.lower()]
                    if len(filtered_options) == 1:  # Only "None" remains
                        st.info(f"No {part} equipment found matching '{search_term}'")
                else:
                    filtered_options = options
                
                # Equipment selector
                selected = st.selectbox(
                    f"{part}:",
                    filtered_options,
                    key=f"select_{part}",
                    help=f"Select {part} equipment"
                )
                
                # Add a divider between parts
                st.divider()

    with col2:
        st.subheader("📊 Equipment Stats")
        
        # Calculate total stats
        total_attack = 0
        total_defense = 0
        skills = defaultdict(int)
        ex_skills = {}

        # Process selected equipment
        equipped_items = 0
        for part in ['Weapon', 'Head', 'Body', 'Hands', 'Feet', 'Acc']:
            selected = st.session_state[f"select_{part}"]
            if selected != "None":
                equipped_items += 1
                # Find matching equipment
                for equip in st.session_state.database.equipment_by_part[part]:
                    if format_equipment_name(equip) == selected:
                        total_attack += equip.attack
                        total_defense += equip.defense
                        
                        # Add regular skills
                        for skill, boost in equip.skills.items():
                            skills[skill] += boost
                        
                        # Add EX skills (take highest boost)
                        if equip.ex_skill:
                            skill_name, boost = equip.ex_skill
                            if skill_name not in ex_skills or boost > ex_skills[skill_name]:
                                ex_skills[skill_name] = boost
                        break

        # Display stats with metrics
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Attack", total_attack)
        with col2b:
            st.metric("Defense", total_defense)
        
        st.metric("Equipped Items", f"{equipped_items}/6")
        
        if skills:
            st.write("💫 Skills:")
            for skill, boost in sorted(skills.items()):
                st.write(f"- {skill}: +{boost}")
        else:
            st.info("No skills from equipped items")
        
        if ex_skills:
            st.write("✨ EX Skills:")
            for skill, boost in sorted(ex_skills.items()):
                st.write(f"- {skill}: R{boost}")
        else:
            st.info("No EX skills from equipped items")

    # Add search functionality
    st.sidebar.subheader("Search Equipment")
    
    # Create a container for search results in the main area
    search_results_container = st.empty()
    
    # Skill requirements
    st.sidebar.write("Required Skills:")
    required_skills = {}
    skill_list = sorted(list(st.session_state.database.all_skills))
    
    num_skills = st.sidebar.number_input("Number of skills to search for", min_value=0, max_value=5, value=1)
    
    for i in range(num_skills):
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            skill = st.selectbox(f"Skill {i+1}", ["None"] + skill_list, key=f"search_skill_{i}")
        with col2:
            level = st.number_input(f"Min Level {i+1}", min_value=0, max_value=20, value=1, key=f"search_level_{i}")
        if skill != "None":
            required_skills[skill] = level

    # EX Skill requirements
    st.sidebar.write("Required EX Skills:")
    required_ex_skills = set()
    num_ex_skills = st.sidebar.number_input("Number of EX skills to search for", min_value=0, max_value=3, value=0)
    
    for i in range(num_ex_skills):
        ex_skill = st.sidebar.selectbox(f"EX Skill {i+1}", ["None"] + skill_list, key=f"search_ex_skill_{i}")
        if ex_skill != "None":
            required_ex_skills.add(ex_skill)

    # Stat requirements
    st.sidebar.write("Stat Requirements:")
    min_attack = st.sidebar.number_input("Minimum Attack", min_value=0, value=0)
    min_defense = st.sidebar.number_input("Minimum Defense", min_value=0, value=0)
    
    # Gender requirement
    gender = st.sidebar.selectbox("Gender Requirement", ["All", "Male", "Female"])

    # Search button
    if st.sidebar.button("Search Combinations"):
        try:
            with st.spinner("Searching for equipment combinations..."):
                # Show search criteria
                search_results_container.subheader("Search Criteria")
                if required_skills:
                    search_results_container.write("Required Skills:")
                    for skill, level in required_skills.items():
                        search_results_container.write(f"- {skill}: minimum +{level}")
                if required_ex_skills:
                    search_results_container.write("Required EX Skills:")
                    for skill in required_ex_skills:
                        search_results_container.write(f"- {skill}")
                if min_attack > 0 or min_defense > 0:
                    search_results_container.write("Minimum Stats:")
                    if min_attack > 0:
                        search_results_container.write(f"- Attack: {min_attack}")
                    if min_defense > 0:
                        search_results_container.write(f"- Defense: {min_defense}")
                if gender != "All":
                    search_results_container.write(f"Gender: {gender}")
                
                requirements = EquipmentRequirement(
                    required_skills=required_skills,
                    required_ex_skills=required_ex_skills,
                    min_attack=min_attack,
                    min_defense=min_defense,
                    gender=gender
                )
                
                # Find combinations
                combinations = find_combinations(st.session_state.database, requirements)
                
                # Display results
                search_results_container.subheader("Search Results")
                if not combinations:
                    search_results_container.warning("No combinations found matching the requirements.")
                else:
                    search_results_container.success(f"Found {len(combinations)} combinations matching your requirements!")
                    
                    # Create columns for the results
                    cols = st.columns(2)
                    for i, combo in enumerate(combinations, 1):
                        with cols[i % 2].expander(
                            f"Combination {i} - Attack: {combo.total_attack}, Defense: {combo.total_defense}"
                            f" (Score: {combo.contribution_score:.2f})"
                        ):
                            # Store the combination in session state
                            combo_key = f"combo_{i}"
                            st.session_state[combo_key] = combo.equipment
                            
                            for slot, equip_name in combo.equipment.items():
                                if equip_name != "None":
                                    st.write(f"{slot}: {equip_name}")
                            st.write("\nSkills:")
                            for skill, level in combo.skills.items():
                                st.write(f"- {skill}: +{level}")
                            if combo.ex_skills:
                                st.write("\nEX Skills:")
                                for skill, level in combo.ex_skills.items():
                                    st.write(f"- {skill}: R{level}")
                            
                            # Add button to apply this combination
                            if st.button("Apply This Combination", key=f"apply_{i}"):
                                # Apply the combination from session state
                                for slot, equip_name in st.session_state[combo_key].items():
                                    st.session_state[f"select_{slot}"] = equip_name
                                st.experimental_rerun()
        except Exception as e:
            search_results_container.error(f"An error occurred while searching: {str(e)}")
            st.error("Please try different search criteria or contact the administrator if the problem persists.")

@dataclass
class EquipmentCombination:
    equipment: Dict[str, str]  # slot: equipment_name
    total_attack: int
    total_defense: int
    skills: Dict[str, int]
    ex_skills: Dict[str, int]
    contribution_score: float = 0.0  # Added contribution score

def calculate_contribution_score(equipment: Equipment, requirements: EquipmentRequirement) -> float:
    """Calculate how much this equipment contributes to meeting requirements."""
    score = 0.0
    
    # Add contribution from attack and defense if they're required
    if requirements.min_attack > 0:
        score += float(equipment.attack) / requirements.min_attack
        
    if requirements.min_defense > 0:
        score += float(equipment.defense) / requirements.min_defense
        
    # Add contribution from skills
    for skill_name, required_level in requirements.required_skills.items():
        if skill_name in equipment.skills:
            score += float(equipment.skills[skill_name]) / required_level
            
    # Add contribution from EX skills
    if equipment.ex_skill and equipment.ex_skill[0] in requirements.required_ex_skills:
        score += float(equipment.ex_skill[1])  # EX skills weighted more heavily
                
    return score

def sort_equipment_by_score(equipment_list: List[Equipment], requirements: EquipmentRequirement) -> List[Equipment]:
    """Sort equipment by their contribution score."""
    scored_equipment = [(e, calculate_contribution_score(e, requirements)) for e in equipment_list]
    scored_equipment.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scored_equipment]

@st.cache_data(show_spinner=False)
def find_combinations(_database: EquipmentDatabase, requirements: EquipmentRequirement) -> List[EquipmentCombination]:
    """Find equipment combinations that meet the given requirements."""
    valid_combinations = []
    total_checked = 0
    
    # Get all equipment that matches gender requirement and sort by contribution score
    valid_equipment = defaultdict(list)
    for part, equipment_list in _database.equipment_by_part.items():
        valid_pieces = []
        for equip in equipment_list:
            if requirements.gender == "All" or equip.gender in [requirements.gender, "All"]:
                valid_pieces.append(equip)
        valid_equipment[part] = sort_equipment_by_score(valid_pieces, requirements)
    
    # Calculate total possible combinations
    total_combinations = 1
    for equip_list in valid_equipment.values():
        total_combinations *= (len(equip_list) + 1)  # +1 for empty slot option
    
    def check_continue() -> bool:
        """Check if we should continue searching based on progress."""
        if len(valid_combinations) >= 10:
            if not st.session_state.get('continue_search', True):
                return False
            st.session_state.continue_search = st.sidebar.button(
                "Found 10 combinations. Continue searching?",
                key="continue_10"
            )
            return st.session_state.continue_search
            
        if total_checked >= 10000:
            if not st.session_state.get('continue_search', True):
                return False
            st.session_state.continue_search = st.sidebar.button(
                f"Checked {total_checked} combinations. Continue searching?",
                key="continue_10000"
            )
            return st.session_state.continue_search
            
        return True
    
    # Helper function to check if a combination meets requirements
    def meets_requirements(combination: Dict[str, Equipment]) -> bool:
        total_attack = sum(e.attack for e in combination.values() if e is not None)
        total_defense = sum(e.defense for e in combination.values() if e is not None)
        
        # Early exit if stats don't meet requirements
        if total_attack < requirements.min_attack or total_defense < requirements.min_defense:
            return False
        
        # Calculate total skills
        skills = defaultdict(int)
        ex_skills = {}
        
        # Track required skills found
        required_skills_found = set()
        required_ex_skills_found = set()
        
        for equip in combination.values():
            if equip is not None:
                # Add regular skills
                for skill, boost in equip.skills.items():
                    skills[skill] += boost
                    if skill in requirements.required_skills:
                        if skills[skill] >= requirements.required_skills[skill]:
                            required_skills_found.add(skill)
                
                # Add EX skills (take highest boost)
                if equip.ex_skill:
                    skill_name, boost = equip.ex_skill
                    if skill_name not in ex_skills or boost > ex_skills[skill_name]:
                        ex_skills[skill_name] = boost
                        if skill_name in requirements.required_ex_skills:
                            required_ex_skills_found.add(skill_name)
                
                # Early exit if we found all required skills
                if (len(required_skills_found) == len(requirements.required_skills) and
                    len(required_ex_skills_found) == len(requirements.required_ex_skills)):
                    return True
        
        # Check required skills
        for skill, required_level in requirements.required_skills.items():
            if skills.get(skill, 0) < required_level:
                return False
        
        # Check required EX skills
        for skill in requirements.required_ex_skills:
            if skill not in ex_skills:
                return False
        
        return True
    
    # Generate combinations using recursive approach with early pruning
    current_combination = {
        'Weapon': None,
        'Head': None,
        'Body': None,
        'Hands': None,
        'Feet': None,
        'Acc': None
    }
    
    def recursive_search(slot_index: int = 0):
        nonlocal total_checked
        
        if not check_continue():
            return
            
        slots = list(current_combination.keys())
        if slot_index >= len(slots):
            if meets_requirements(current_combination):
                # Create EquipmentCombination object
                equipment_names = {
                    slot: format_equipment_name(equip) for slot, equip in current_combination.items()
                }
                total_attack = sum(e.attack for e in current_combination.values() if e is not None)
                total_defense = sum(e.defense for e in current_combination.values() if e is not None)
                
                # Calculate skills and contribution score
                skills = defaultdict(int)
                ex_skills = {}
                contribution_score = 0.0
                
                for equip in current_combination.values():
                    if equip is not None:
                        contribution_score += calculate_contribution_score(equip, requirements)
                        for skill, boost in equip.skills.items():
                            skills[skill] += boost
                        if equip.ex_skill:
                            skill_name, boost = equip.ex_skill
                            if skill_name not in ex_skills or boost > ex_skills[skill_name]:
                                ex_skills[skill_name] = boost
                
                combo = EquipmentCombination(
                    equipment=equipment_names,
                    total_attack=total_attack,
                    total_defense=total_defense,
                    skills=dict(skills),
                    ex_skills=ex_skills,
                    contribution_score=contribution_score
                )
                valid_combinations.append(combo)
            total_checked += 1
            return
        
        current_slot = slots[slot_index]
        
        # Try each equipment piece for current slot (already sorted by contribution)
        for equipment in valid_equipment[current_slot]:
            current_combination[current_slot] = equipment
            recursive_search(slot_index + 1)
            if len(valid_combinations) >= 10 and not st.session_state.get('continue_search', True):
                return
        
        # Also try without any equipment in this slot
        current_combination[current_slot] = None
        recursive_search(slot_index + 1)
    
    recursive_search()
    
    # Sort combinations by contribution score and then total attack + defense
    valid_combinations.sort(key=lambda x: (x.contribution_score, x.total_attack + x.total_defense), reverse=True)
    return valid_combinations

if __name__ == "__main__":
    main() 