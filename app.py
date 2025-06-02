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
        self.equipment_by_part = defaultdict(list)
        self.all_skills = set()
        
    def load_from_csv(self, df: pd.DataFrame):
        for _, row in df.iterrows():
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
                    except (ValueError, AttributeError):
                        continue
            
            # Parse EX skill
            ex_skill = None
            if pd.notna(row['EX_Skill']):
                try:
                    ex_boost = int(float(row['Boost_Ex'])) if pd.notna(row['Boost_Ex']) else 0
                    ex_skill = (str(row['EX_Skill']).strip(), ex_boost)
                    self.all_skills.add(str(row['EX_Skill']))
                except (ValueError, AttributeError):
                    pass
            
            # Parse released status
            released = "Yes"  # Default
            if pd.notna(row['Released']):
                released = str(row['Released']).strip()
            
            try:
                equipment = Equipment(
                    part=str(row['Part']).strip(),
                    name=str(row['Name']).strip(),
                    attack=int(float(row['Att'])) if pd.notna(row['Att']) else 0,
                    defense=int(float(row['Def'])) if pd.notna(row['Def']) else 0,
                    skills=skills,
                    ex_skill=ex_skill,
                    gender=str(row['Male/Female/All']).strip(),
                    released=released
                )
                self.equipment_by_part[equipment.part].append(equipment)
            except (ValueError, KeyError, AttributeError) as e:
                st.error(f"Error processing row: {row}")
                st.error(f"Error: {e}")
                continue

def format_equipment_name(equipment: Equipment) -> str:
    if equipment is None:
        return "None"
    
    parts = [f"{equipment.name}     "]
    
    skill_parts = []
    for skill, boost in equipment.skills.items():
        skill_parts.append(f"{skill}+{boost}")
    if skill_parts:
        parts.append("(" + ", ".join(skill_parts) + ")")
    
    if equipment.ex_skill:
        skill_name, boost = equipment.ex_skill
        parts.append(f" (EX: {skill_name} R{boost})")
    
    return "".join(parts)

# Add authentication
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
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
    st.set_page_config(page_title="GVO Equipment Analyzer", layout="wide")
    st.title("GVO Equipment Analyzer")

    if not check_password():
        st.stop()  # Do not continue if password is incorrect
        return

    # Initialize session state
    if 'database' not in st.session_state:
        st.session_state.database = EquipmentDatabase()
        try:
            # Try to get CSV data from secrets
            csv_content = st.secrets["equipment_csv"]
            # Convert base64 string back to CSV data
            csv_data = base64.b64decode(csv_content).decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_data))
            st.session_state.database.load_from_csv(df)
            st.session_state.df = df
        except Exception as e:
            st.error("Error loading equipment data. Please contact the administrator.")
            st.stop()
            return

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Equipment Selection")
        
        # Equipment selection for each part
        for part in ['Weapon', 'Head', 'Body', 'Hands', 'Feet', 'Acc']:
            equipment_list = st.session_state.database.equipment_by_part[part]
            
            # Create equipment options
            options = ["None"] + [format_equipment_name(e) for e in equipment_list]
            
            # Equipment selector
            selected = st.selectbox(
                f"{part}:",
                options,
                key=f"select_{part}"
            )

    with col2:
        st.subheader("Equipment Stats")
        
        # Calculate total stats
        total_attack = 0
        total_defense = 0
        skills = defaultdict(int)
        ex_skills = {}

        # Process selected equipment
        for part in ['Weapon', 'Head', 'Body', 'Hands', 'Feet', 'Acc']:
            selected = st.session_state[f"select_{part}"]
            if selected != "None":
                # Find matching equipment
                for equip in st.session_state.database.equipment_by_part[part]:
                    if format_equipment_name(equip) == selected:
                        total_attack += equip.attack
                        total_defense += equip.defense
                        
                        # Add regular skills
                        for skill, boost in equip.skills.items():
                            skills[skill] += boost
                        
                        # Add EX skill
                        if equip.ex_skill:
                            skill_name, boost = equip.ex_skill
                            if skill_name in ex_skills:
                                ex_skills[skill_name] = max(ex_skills[skill_name], boost)
                            else:
                                ex_skills[skill_name] = boost
                        break

        # Display stats
        st.metric("Attack", total_attack)
        st.metric("Defense", total_defense)
        
        if skills:
            st.subheader("Skills")
            for skill, boost in sorted(skills.items()):
                st.text(f"{skill}: +{boost}")
        
        if ex_skills:
            st.subheader("EX Skills")
            for skill, boost in sorted(ex_skills.items()):
                st.text(f"{skill}: R{boost}")

    # Add Find Combinations section
    st.subheader("Find Combinations")
    with st.expander("Find Equipment Combinations"):
        # Skill requirements
        st.write("Skill Requirements")
        skill_cols = st.columns(3)
        required_skills = {}
        
        all_skills = sorted(list(st.session_state.database.all_skills))
        for i, col in enumerate(skill_cols):
            with col:
                skill = st.selectbox(f"Skill {i+1}", [""] + all_skills, key=f"req_skill_{i}")
                if skill:
                    level = st.number_input(f"Min Level {i+1}", min_value=0, key=f"req_level_{i}")
                    if level > 0:
                        required_skills[skill] = level

        # Basic requirements
        st.write("Basic Requirements")
        req_cols = st.columns(3)
        with req_cols[0]:
            min_attack = st.number_input("Min Attack", min_value=0)
        with req_cols[1]:
            min_defense = st.number_input("Min Defense", min_value=0)
        with req_cols[2]:
            gender = st.selectbox("Gender", ["All", "Male", "Female"])

        # Include not-released checkbox
        include_unreleased = st.checkbox("Include not-released")

        if st.button("Find Combinations"):
            # Create requirements object
            requirements = EquipmentRequirement(
                required_skills=required_skills,
                required_ex_skills=set(),
                min_attack=min_attack,
                min_defense=min_defense,
                gender=gender
            )

            # TODO: Implement combination finding logic
            st.info("Combination finding will be implemented in the next update!")

if __name__ == "__main__":
    main() 