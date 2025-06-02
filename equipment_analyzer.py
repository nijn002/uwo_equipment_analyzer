import csv
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, messagebox
import os
import itertools
import sys

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

@dataclass
class EquipmentRequirement:
    required_skills: Dict[str, int]  # skill_name: minimum_level
    required_ex_skills: Set[str]
    min_attack: int = 0
    min_defense: int = 0
    gender: str = "All"

@dataclass
class EquipmentCombination:
    equipment: Dict[str, str]  # slot: equipment_name
    total_attack: int
    total_defense: int
    skills: Dict[str, int]
    ex_skills: Dict[str, int]  # Changed to Dict to store boost values
    
    def __str__(self):
        return (
            f"Equipment: {self.equipment}\n"
            f"Attack: {self.total_attack}, Defense: {self.total_defense}\n"
            f"Skills: {dict(self.skills)}\n"
            f"EX Skills: {dict(self.ex_skills)}"
        )

class Equipment:
    def __init__(self, part: str, name: str, attack: int, defense: int, 
                 skills: Dict[str, int], ex_skill: Tuple[str, int], gender: str, released: str = "Yes"):
        self.part = part
        self.name = name
        self.attack = attack
        self.defense = defense
        self.skills = skills  # Regular skills and their boosts
        self.ex_skill = ex_skill  # (skill_name, boost)
        self.gender = gender
        self.released = released  # "Yes", "Japan"

class EquipmentSet:
    def __init__(self):
        self.parts = {
            'Weapon': None,
            'Head': None,
            'Body': None,
            'Hands': None,
            'Feet': None,
            'Acc': None
        }
        
    def add_equipment(self, equipment: Equipment) -> bool:
        if equipment.part in self.parts:
            self.parts[equipment.part] = equipment
            return True
        return False
    
    def remove_equipment(self, part: str) -> bool:
        if part in self.parts:
            self.parts[part] = None
            return True
        return False
    
    def calculate_totals(self) -> Dict:
        total_stats = {
            'Attack': 0,
            'Defense': 0,
            'Skills': defaultdict(int),
            'EX_Skills': defaultdict(list)  # Track all EX skill boosts
        }
        
        for equip in self.parts.values():
            if equip is not None:
                # Add basic stats
                total_stats['Attack'] += equip.attack
                total_stats['Defense'] += equip.defense
                
                # Add regular skill boosts (these stack)
                for skill, boost in equip.skills.items():
                    total_stats['Skills'][skill] += boost
                
                # Add EX skill to list (we'll take highest later)
                if equip.ex_skill:
                    skill_name, boost = equip.ex_skill
                    total_stats['EX_Skills'][skill_name].append(boost)
        
        # Process EX skills - take highest boost for each skill
        for skill, boosts in total_stats['EX_Skills'].items():
            if boosts:  # If we have any boosts for this skill
                total_stats['Skills'][f"{skill} (EX)"] = max(boosts)
        
        del total_stats['EX_Skills']  # Remove temporary EX skills tracking
        return total_stats

class EquipmentDatabase:
    def __init__(self):
        self.equipment_by_part = defaultdict(list)
        self.all_skills = set()
        
    def load_from_csv(self, filename: str):
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse skills and boosts
                skills = {}
                # Handle the typo in the header (Skill_2 appears twice instead of Skill_3)
                skill_columns = {
                    1: ('Skill_1', 'Boost_1'),
                    2: ('Skill_2', 'Boost_2'),
                    3: ('Skill_3', 'Boost_3')  # The second Skill_2 column is actually Skill_3
                }
                
                for i, (skill_col, boost_col) in skill_columns.items():
                    skill_name = row[skill_col].strip() if skill_col in row else ''
                    boost_value = row[boost_col] if boost_col in row else '0'
                    if skill_name:
                        try:
                            boost = int(boost_value.strip())
                            skills[skill_name] = boost
                            self.all_skills.add(skill_name)
                        except (ValueError, AttributeError):
                            continue
                
                # Parse EX skill
                ex_skill = None
                if 'EX_Skill' in row and row['EX_Skill'].strip():
                    try:
                        ex_boost = int(row.get('Boost_Ex', '0').strip())
                        ex_skill = (row['EX_Skill'].strip(), ex_boost)
                        self.all_skills.add(row['EX_Skill'])
                    except (ValueError, AttributeError):
                        pass
                
                # Parse released status
                released = "Yes"  # Default to "Yes" if not specified
                if 'Released' in row:
                    released_value = row['Released'].strip()
                    if released_value:
                        released = released_value
                
                # Create equipment object
                try:
                    # Get the part name, handling possible BOM character
                    part_key = 'Part'
                    if '\ufeffPart' in row:
                        part_key = '\ufeffPart'
                    
                    equipment = Equipment(
                        part=row[part_key].strip(),
                        name=row['Name'].strip(),
                        attack=int(row['Att']) if row.get('Att', '').strip() else 0,
                        defense=int(row['Def']) if row.get('Def', '').strip() else 0,
                        skills=skills,
                        ex_skill=ex_skill,
                        gender=row['Male/Female/All'].strip(),
                        released=released
                    )
                    self.equipment_by_part[equipment.part].append(equipment)
                except (ValueError, KeyError, AttributeError) as e:
                    print(f"Error processing row: {row}")
                    print(f"Error: {e}")
                    continue

    def find_combinations(self, requirements: Dict[str, int]) -> List[EquipmentSet]:
        """Find equipment combinations that meet the given requirements"""
        # TODO: Implement combination search algorithm
        pass

class FindWindow:
    def __init__(self, parent, part, database, on_select_callback):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Find {part} Equipment")
        self.window.geometry("800x500")
        self.window.transient(parent)
        self.window.grab_set()
        
        self.database = database
        self.part = part
        self.on_select_callback = on_select_callback
        
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create filter frame for skills, gender, and release status
        filter_frame = ttk.Frame(main_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create skill selectors frame
        skills_frame = ttk.LabelFrame(filter_frame, text="Skill Criteria", padding="5")
        skills_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Create filter frame for gender and release status
        right_filters_frame = ttk.Frame(filter_frame)
        right_filters_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create gender filter frame
        gender_frame = ttk.LabelFrame(right_filters_frame, text="Gender Filter", padding="5")
        gender_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Add gender checkboxes
        self.male_var = tk.BooleanVar(value=False)
        self.female_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(gender_frame, text="Male", variable=self.male_var, 
                       command=self._update_equipment_list).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(gender_frame, text="Female", variable=self.female_var,
                       command=self._update_equipment_list).pack(anchor=tk.W, pady=2)
        
        # Create release status filter frame
        release_frame = ttk.LabelFrame(right_filters_frame, text="Release Filter", padding="5")
        release_frame.pack(fill=tk.X)
        
        # Add release status checkbox with updated text
        self.include_unreleased_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(release_frame, text="Include not-released", 
                       variable=self.include_unreleased_var,
                       command=self._update_equipment_list).pack(anchor=tk.W, pady=2)
        
        # Get all unique skills, separated by regular and EX
        all_regular_skills = sorted([skill for skill in database.all_skills 
                                   if not any(equip.ex_skill and equip.ex_skill[0] == skill 
                                            for part_equips in database.equipment_by_part.values()
                                            for equip in part_equips)])
        
        all_ex_skills = sorted([skill for skill in database.all_skills 
                              if any(equip.ex_skill and equip.ex_skill[0] == skill 
                                    for part_equips in database.equipment_by_part.values()
                                    for equip in part_equips)])
        
        # Create regular skill selectors
        self.skill_vars = []
        for i in range(3):
            skill_frame = ttk.Frame(skills_frame)
            skill_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(skill_frame, text=f"Skill {i+1}:", width=10).pack(side=tk.LEFT, padx=(0, 10))
            
            var = tk.StringVar()
            skill_dropdown = ttk.Combobox(skill_frame, textvariable=var, state='readonly', width=30)
            skill_dropdown['values'] = [''] + all_regular_skills
            skill_dropdown.set('')
            skill_dropdown.pack(side=tk.LEFT)
            
            self.skill_vars.append(var)
        
        # Create EX skill selector
        ex_frame = ttk.Frame(skills_frame)
        ex_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(ex_frame, text="EX Skill:", width=10).pack(side=tk.LEFT, padx=(0, 10))
        
        self.ex_skill_var = tk.StringVar()
        ex_skill_dropdown = ttk.Combobox(ex_frame, textvariable=self.ex_skill_var, state='readonly', width=30)
        ex_skill_dropdown['values'] = [''] + all_ex_skills
        ex_skill_dropdown.set('')
        ex_skill_dropdown.pack(side=tk.LEFT)
        
        # Create equipment list
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview for equipment list
        self.tree = ttk.Treeview(list_frame, columns=('boosts', 'skills'), show='headings')
        self.tree.heading('boosts', text='Boosts')
        self.tree.heading('skills', text='Equipment & Skills')
        
        # Configure column widths
        self.tree.column('boosts', width=100, anchor='center')
        self.tree.column('skills', width=650)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        add_button = ttk.Button(button_frame, text="Add Selected", command=self._on_add)
        add_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.window.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Bind events
        for var in self.skill_vars:
            var.trace('w', lambda *args: self._update_equipment_list())
        self.ex_skill_var.trace('w', lambda *args: self._update_equipment_list())
        self.tree.bind('<Double-Button-1>', lambda e: self._on_add())
        
        # Initial population of list
        self._update_equipment_list()
        
    def _update_equipment_list(self):
        """Update the equipment list based on selected skills, gender, and release status"""
        # Clear current items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get selected skills (non-empty)
        selected_skills = [var.get() for var in self.skill_vars if var.get()]
        selected_ex_skill = self.ex_skill_var.get()
        
        # Get gender filter state
        male_checked = self.male_var.get()
        female_checked = self.female_var.get()
        
        # Get release filter state
        include_unreleased = self.include_unreleased_var.get()
        
        # Get equipment for the part
        equipment_list = self.database.equipment_by_part[self.part]
        
        # Filter and sort equipment
        filtered_equipment = []
        for equip in equipment_list:
            # Check release status
            if not include_unreleased and equip.released != "Yes":
                continue
                
            # Check if equipment has all selected regular skills
            has_all_skills = all(skill in equip.skills for skill in selected_skills)
            
            # Check if equipment has the selected EX skill
            has_ex_skill = (not selected_ex_skill) or (
                equip.ex_skill and equip.ex_skill[0] == selected_ex_skill
            )
            
            # Check gender requirement
            meets_gender = (
                # Show all if both checked or both unchecked
                (not male_checked and not female_checked) or
                (male_checked and female_checked) or
                # Show if matches gender selection
                (male_checked and equip.gender in ['Male', 'All']) or
                (female_checked and equip.gender in ['Female', 'All'])
            )
            
            if has_all_skills and has_ex_skill and meets_gender:
                # Calculate total boost value for sorting
                total_boost = sum(equip.skills.get(skill, 0) for skill in selected_skills)
                if selected_ex_skill and equip.ex_skill and equip.ex_skill[0] == selected_ex_skill:
                    total_boost += equip.ex_skill[1]
                
                filtered_equipment.append((total_boost, equip))
        
        # Sort by total boost (descending), then name (ascending)
        filtered_equipment.sort(key=lambda x: (-x[0], x[1].name))
        
        # Add to tree
        for total_boost, equip in filtered_equipment:
            # Create boost display string
            boosts = []
            for skill in selected_skills:
                if skill in equip.skills:
                    boosts.append(f"{skill}: +{equip.skills[skill]}")
            if selected_ex_skill and equip.ex_skill and equip.ex_skill[0] == selected_ex_skill:
                boosts.append(f"{selected_ex_skill}: R{equip.ex_skill[1]}")
            
            boost_display = ", ".join(boosts) if boosts else ""
            display_name = self._format_equipment_name(equip)
            self.tree.insert('', 'end', values=(boost_display, display_name))
    
    def _format_equipment_name(self, equipment: Equipment) -> str:
        """Format equipment name with skills for display"""
        if equipment is None:
            return "None"
        
        parts = [f"{equipment.name}     "]
        
        # Add regular skills
        skill_parts = []
        for skill, boost in equipment.skills.items():
            skill_parts.append(f"{skill}+{boost}")
        if skill_parts:
            parts.append("(" + ", ".join(skill_parts) + ")")
        
        # Add EX skill if present
        if equipment.ex_skill:
            skill_name, boost = equipment.ex_skill
            parts.append(f" (EX: {skill_name} R{boost})")
        
        return "".join(parts)
    
    def _on_add(self):
        """Handle add button click"""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            equipment_name = item['values'][1]  # Get the full equipment name
            self.on_select_callback(equipment_name)
            self.window.destroy()

class EquipmentOptimizer:
    def __init__(self, csv_path: str):
        # Handle possible BOM in CSV
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.equipment_by_slot = self._organize_by_slot()
        self.skill_columns = [f'Skill_{i}' for i in range(1, 4)]
        self.boost_columns = [f'Boost_{i}' for i in range(1, 4)]
        
    def _organize_by_slot(self) -> Dict[str, pd.DataFrame]:
        """Organize equipment data by slot type."""
        return {slot: self.df[self.df['Part'] == slot] for slot in self.df['Part'].unique()}
    
    def _is_strictly_dominated(self, equip1_row, equip2_row, requirements: EquipmentRequirement) -> bool:
        """
        Check if equip1 is strictly dominated by equip2 for the given requirements.
        Only considers relevant attributes based on requirements.
        """
        # Helper function to safely get numeric value
        def safe_numeric(value):
            if pd.isna(value):
                return 0
            try:
                return float(str(value).strip() or '0')
            except (ValueError, TypeError):
                return 0
                
        # Helper function to safely get string value
        def safe_string(value):
            if pd.isna(value):
                return ''
            return str(value).strip()

        # If they're for different slots or genders, no dominance
        if safe_string(equip1_row['Part']) != safe_string(equip2_row['Part']):
            return False
        if safe_string(equip1_row['Male/Female/All']) != safe_string(equip2_row['Male/Female/All']):
            return False
            
        # Check attack/defense if they're part of requirements
        if requirements.min_attack > 0:
            att1 = safe_numeric(equip1_row['Att'])
            att2 = safe_numeric(equip2_row['Att'])
            if att2 < att1:
                return False
                
        if requirements.min_defense > 0:
            def1 = safe_numeric(equip1_row['Def'])
            def2 = safe_numeric(equip2_row['Def'])
            if def2 < def1:
                return False
        
        # Check required skills
        any_better = False
        for skill_name in requirements.required_skills:
            boost1 = 0
            boost2 = 0
            
            # Check all skill columns for the required skill
            for i in range(1, 4):
                skill1 = safe_string(equip1_row[f'Skill_{i}'])
                skill2 = safe_string(equip2_row[f'Skill_{i}'])
                
                if skill1 == skill_name:
                    b1 = safe_numeric(equip1_row[f'Boost_{i}'])
                    boost1 = max(boost1, b1)
                if skill2 == skill_name:
                    b2 = safe_numeric(equip2_row[f'Boost_{i}'])
                    boost2 = max(boost2, b2)
            
            if boost2 < boost1:  # If equip2 is worse in any required skill
                return False
            if boost2 > boost1:
                any_better = True
        
        # Check EX skills if required
        if requirements.required_ex_skills:
            ex1 = safe_string(equip1_row['EX_Skill'])
            ex2 = safe_string(equip2_row['EX_Skill'])
            
            # If they have different EX skills and either is required, no dominance
            if ex1 != ex2 and (ex1 in requirements.required_ex_skills or 
                              ex2 in requirements.required_ex_skills):
                return False
            
            # If same required EX skill, check boost
            if ex1 == ex2 and ex1 in requirements.required_ex_skills:
                boost1 = safe_numeric(equip1_row['Boost_Ex'])
                boost2 = safe_numeric(equip2_row['Boost_Ex'])
                if boost2 < boost1:
                    return False
                if boost2 > boost1:
                    any_better = True
        
        return any_better  # Must be better in at least one relevant attribute

    def _get_compatible_equipment(self, slot: str, requirements: EquipmentRequirement) -> pd.DataFrame:
        """Get compatible equipment for a slot, removing dominated equipment."""
        slot_df = self.equipment_by_slot[slot]
        
        print(f"\nDebug - Processing slot {slot}")
        print(f"Initial equipment count: {len(slot_df)}")
        
        # Apply basic filters first
        if requirements.gender != "All":
            slot_df = slot_df[
                (slot_df['Male/Female/All'] == requirements.gender) | 
                (slot_df['Male/Female/All'] == 'All')
            ]
            print(f"After gender filter: {len(slot_df)} items")

        # Pre-filter by skills
        if requirements.required_skills:
            has_required_skill = pd.Series(False, index=slot_df.index)
            
            for skill_name in requirements.required_skills:
                print(f"\nLooking for equipment with {skill_name}")
                for i in range(1, 4):
                    skill_col = f'Skill_{i}'
                    boost_col = f'Boost_{i}'
                    
                    boosts = pd.to_numeric(slot_df[boost_col].fillna('').astype(str).str.strip(), 
                                         errors='coerce').fillna(0)
                    
                    matches = slot_df[skill_col].fillna('').astype(str).str.strip() == skill_name
                    has_boost = boosts > 0
                    
                    has_this_skill = matches & has_boost
                    has_required_skill |= has_this_skill
            
            slot_df = slot_df[has_required_skill]
            print(f"\nAfter skill filter: {len(slot_df)} items")
            
        # Remove dominated equipment
        if len(slot_df) > 1:  # Only check if we have multiple items
            dominated = set()
            for idx1, row1 in slot_df.iterrows():
                if idx1 in dominated:
                    continue
                for idx2, row2 in slot_df.iterrows():
                    if idx1 != idx2 and idx2 not in dominated:
                        if self._is_strictly_dominated(row1, row2, requirements):
                            dominated.add(idx1)
                            break
            
            if dominated:
                slot_df = slot_df.drop(index=dominated)
                print(f"Removed {len(dominated)} dominated items, {len(slot_df)} items remaining")
        
        return slot_df
    
    def _calculate_combination_stats(self, equipment_names: Dict[str, str]) -> Optional[EquipmentCombination]:
        """Calculate stats for a specific equipment combination."""
        if not equipment_names:  # Don't process empty combinations
            return None
            
        total_attack = 0
        total_defense = 0
        skills = defaultdict(int)
        ex_skills = {}
        
        print(f"Debug - Calculating stats for equipment: {equipment_names}")  # Debug print
        
        # Get equipment details from dataframe
        for slot, name in equipment_names.items():
            item = self.df[(self.df['Part'] == slot) & (self.df['Name'] == name)].iloc[0]
            
            # Add attack and defense
            total_attack += item['Att'] if pd.notna(item['Att']) else 0
            total_defense += item['Def'] if pd.notna(item['Def']) else 0
            
            # Add skills
            for skill_col, boost_col in zip(self.skill_columns, self.boost_columns):
                if pd.notna(item[skill_col]) and pd.notna(item[boost_col]):
                    skill_name = str(item[skill_col]).strip()
                    if skill_name:  # Only process if skill name exists
                        try:
                            boost_value = int(float(item[boost_col]))
                            # Add the boost value to the total for this skill
                            skills[skill_name] += boost_value
                        except (ValueError, TypeError):
                            continue
            
            # Add EX skill
            if pd.notna(item['EX_Skill']):
                ex_skill_name = str(item['EX_Skill']).strip()
                try:
                    ex_boost = int(float(item['Boost_Ex'])) if pd.notna(item['Boost_Ex']) else 0
                    if ex_skill_name in ex_skills:
                        ex_skills[ex_skill_name] = max(ex_skills[ex_skill_name], ex_boost)
                    else:
                        ex_skills[ex_skill_name] = ex_boost
                except (ValueError, TypeError):
                    continue
        
        combination = EquipmentCombination(
            equipment=equipment_names.copy(),  # Make a copy to ensure it's not modified
            total_attack=total_attack,
            total_defense=total_defense,
            skills=dict(skills),
            ex_skills=ex_skills
        )
        
        print(f"Debug - Created combination: {combination}")  # Debug print
        return combination
    
    def _meets_requirements(self, combination: EquipmentCombination, requirements: EquipmentRequirement) -> bool:
        """Check if a combination meets the given requirements."""
        # Check attack and defense requirements
        if combination.total_attack < requirements.min_attack:
            return False
        if combination.total_defense < requirements.min_defense:
            return False
            
        # Check required skills - total across all equipment must meet or exceed requirement
        for skill, required_level in requirements.required_skills.items():
            total_level = combination.skills.get(skill, 0)
            if total_level < required_level:
                print(f"Combination has {skill} +{total_level}, need +{required_level}")
                return False
                
        # Check required EX skills
        for skill in requirements.required_ex_skills:
            if skill not in combination.ex_skills:
                return False
                
        return True
    
    def _calculate_skill_potential(self, slot_equipment: Dict[str, pd.DataFrame], requirements: EquipmentRequirement) -> Dict[str, Dict[str, int]]:
        """Calculate maximum potential contribution for each skill from each slot."""
        skill_potentials = {}
        
        for slot, equipment_df in slot_equipment.items():
            skill_potentials[slot] = defaultdict(int)
            
            for _, item in equipment_df.iterrows():
                # Check regular skills
                for skill_col, boost_col in zip(self.skill_columns, self.boost_columns):
                    if pd.notna(item[skill_col]) and pd.notna(item[boost_col]):
                        skill_name = str(item[skill_col]).strip()
                        if skill_name in requirements.required_skills:
                            try:
                                boost = int(float(item[boost_col]))
                                skill_potentials[slot][skill_name] = max(
                                    skill_potentials[slot][skill_name], boost)
                            except (ValueError, TypeError):
                                continue
                
                # Check EX skills
                if pd.notna(item['EX_Skill']):
                    skill_name = str(item['EX_Skill']).strip()
                    if skill_name in requirements.required_ex_skills:
                        try:
                            boost = int(float(item['Boost_Ex'])) if pd.notna(item['Boost_Ex']) else 0
                            skill_potentials[slot][f"{skill_name} (EX)"] = max(
                                skill_potentials[slot][f"{skill_name} (EX)"], boost)
                        except (ValueError, TypeError):
                            continue
        
        return skill_potentials
    
    def _calculate_contribution_score(self, row, requirements: EquipmentRequirement) -> float:
        """Calculate a score representing how much this equipment contributes to meeting requirements."""
        score = 0.0
        
        # Add contribution from attack and defense if they're required
        if requirements.min_attack > 0:
            score += float(row['Att'] if pd.notna(row['Att']) else 0) / requirements.min_attack
            
        if requirements.min_defense > 0:
            score += float(row['Def'] if pd.notna(row['Def']) else 0) / requirements.min_defense
            
        # Add contribution from skills
        for skill_name, required_level in requirements.required_skills.items():
            max_boost = 0
            # Check all skill columns
            for i in range(1, 4):
                if pd.notna(row[f'Skill_{i}']) and str(row[f'Skill_{i}']).strip() == skill_name:
                    try:
                        boost = int(float(row[f'Boost_{i}']))
                        max_boost = max(max_boost, boost)
                    except (ValueError, TypeError):
                        continue
            
            if max_boost > 0:
                score += float(max_boost) / required_level
                
        # Add contribution from EX skills
        if requirements.required_ex_skills and pd.notna(row['EX_Skill']):
            ex_skill = str(row['EX_Skill']).strip()
            if ex_skill in requirements.required_ex_skills:
                try:
                    boost = int(float(row['Boost_Ex'])) if pd.notna(row['Boost_Ex']) else 0
                    score += float(boost)  # EX skills are weighted more heavily
                except (ValueError, TypeError):
                    pass
                    
        return score
    
    def _can_meet_requirements(self, slot_potentials: Dict[str, Dict[str, int]], 
                             requirements: EquipmentRequirement,
                             used_slots: Set[str]) -> bool:
        """Check if it's still possible to meet requirements with remaining slots."""
        remaining_requirements = requirements.required_skills.copy()
        
        # Subtract what we've already achieved
        for slot in used_slots:
            if slot in slot_potentials:
                for skill, boost in slot_potentials[slot].items():
                    if skill in remaining_requirements:
                        remaining_requirements[skill] = max(0, 
                            remaining_requirements[skill] - boost)
        
        # Check what we can still achieve with unused slots
        unused_slots = set(slot_potentials.keys()) - used_slots
        potential_boosts = defaultdict(int)
        
        for slot in unused_slots:
            for skill, boost in slot_potentials[slot].items():
                if skill in remaining_requirements:
                    potential_boosts[skill] += boost
        
        # Check if we can meet remaining requirements
        for skill, required in remaining_requirements.items():
            if required > 0 and potential_boosts[skill] < required:
                return False
                
        return True
    
    def find_optimal_combinations(self, requirements: EquipmentRequirement, max_results: int = 10, max_checks: int = 10000, 
                               selected_equipment: Dict[str, str] = None):
        """Find optimal equipment combinations that meet the given requirements."""
        # Get compatible equipment for each slot
        slot_equipment = {
            slot: self._get_compatible_equipment(slot, requirements)
            for slot in self.equipment_by_slot.keys()
        }
        
        # Calculate skill potentials for early termination
        skill_potentials = self._calculate_skill_potential(slot_equipment, requirements)
        
        # Sort slots by their potential contribution
        sorted_slots = self._sort_slots_by_priority(slot_equipment, requirements)
        
        # Initialize tracking variables
        valid_combinations = []
        total_checked = 0
        total_combinations = 1
        for slot_df in slot_equipment.values():
            total_combinations *= (len(slot_df) + 1)  # +1 for empty slot option
            
        print(f"Total possible combinations: {total_combinations}")
        
        def recursive_search(depth=0, current_combination=None):
            nonlocal total_checked
            
            if current_combination is None:
                current_combination = {}
                
            # Check if we've hit our limits
            if not self._check_continue(total_checked, len(valid_combinations), total_combinations):
                return
                
            # If we've used all slots, check if this combination works
            if depth >= len(sorted_slots):
                total_checked += 1
                if total_checked % 1000 == 0:
                    print(f"Checked {total_checked} combinations...")
                    
                combination = self._calculate_combination_stats(current_combination)
                if combination and self._meets_requirements(combination, requirements):
                    valid_combinations.append(combination)
                return
                
            current_slot = sorted_slots[depth]
            slot_df = slot_equipment[current_slot]
            
            # If this slot has a selected piece of equipment, try only that one
            if (selected_equipment and current_slot in selected_equipment and 
                any(row['Name'] == selected_equipment[current_slot] for _, row in slot_df.iterrows())):
                current_combination[current_slot] = selected_equipment[current_slot]
                recursive_search(depth + 1, current_combination)
                del current_combination[current_slot]
                return
            
            # Try each piece of equipment for this slot
            equipment_tried = set()
            for _, row in slot_df.iterrows():
                # Skip if we've already tried this equipment (handles duplicates)
                if row['Name'] in equipment_tried:
                    continue
                equipment_tried.add(row['Name'])
                
                # Add this equipment to the combination
                current_combination[current_slot] = row['Name']
                
                # Check if we can still meet requirements with this choice
                if self._can_meet_requirements(skill_potentials, requirements, set(current_combination.keys())):
                    recursive_search(depth + 1, current_combination)
                
                # Remove this equipment for the next iteration
                del current_combination[current_slot]
                
                # If we have enough valid combinations, stop searching
                if len(valid_combinations) >= max_results:
                    return
            
            # Try without any equipment in this slot
            recursive_search(depth + 1, current_combination)
        
        # Start the recursive search
        recursive_search()
        
        # Sort combinations by total contribution
        valid_combinations.sort(key=lambda x: (
            sum(x.skills.get(skill, 0) for skill in requirements.required_skills),
            x.total_attack,
            x.total_defense
        ), reverse=True)
        
        # Show results summary
        self._show_results(valid_combinations, requirements, total_checked, total_combinations)
        
        return valid_combinations[:max_results]
    
    def _check_continue(self, total_checked, valid_found, total_combinations):
        """Check if we should continue searching based on progress."""
        # If we've found enough valid combinations
        if valid_found >= 10:
            return False
            
        # If we've checked too many combinations
        if total_checked >= 10000:
            return False
            
        # If we've checked a reasonable portion of total combinations
        if total_combinations < 1000000:  # For small search spaces
            if total_checked >= total_combinations:
                return False
        else:  # For large search spaces
            if total_checked >= 100000:  # Hard limit for very large spaces
                return False
                
        return True
    
    def _show_results(self, valid_combinations, requirements, total_checked, total_combinations):
        """Display search results and statistics."""
        print("\nSearch Results:")
        print(f"Total combinations possible: {total_combinations}")
        print(f"Combinations checked: {total_checked}")
        print(f"Valid combinations found: {len(valid_combinations)}")
        
        if valid_combinations:
            print("\nBest combination found:")
            best = valid_combinations[0]
            print(f"Equipment: {best.equipment}")
            print(f"Attack: {best.total_attack}")
            print(f"Defense: {best.total_defense}")
            print("Skills:")
            for skill, boost in best.skills.items():
                req = requirements.required_skills.get(skill, 0)
                if req > 0:
                    print(f"  {skill}: +{boost} (required: +{req})")
                else:
                    print(f"  {skill}: +{boost}")
            if best.ex_skills:
                print("EX Skills:")
                for skill, boost in best.ex_skills.items():
                    print(f"  {skill}: +{boost}")
    
    def _identify_required_slots(self, requirements: EquipmentRequirement) -> List[str]:
        """Identify which slots are required to meet the requirements."""
        required_slots = set()
        
        # Check each slot's potential contribution
        for slot, equipment in self.equipment_by_slot.items():
            for _, row in equipment.iterrows():
                # Check if this equipment contributes to requirements
                if requirements.min_attack > 0 and row['Att'] > 0:
                    required_slots.add(slot)
                    continue
                    
                if requirements.min_defense > 0 and row['Def'] > 0:
                    required_slots.add(slot)
                    continue
                    
                # Check regular skills
                for skill_col, boost_col in zip(self.skill_columns, self.boost_columns):
                    if pd.notna(row[skill_col]) and str(row[skill_col]).strip() in requirements.required_skills:
                        if pd.notna(row[boost_col]) and float(row[boost_col]) > 0:
                            required_slots.add(slot)
                            break
                            
                # Check EX skills
                if pd.notna(row['EX_Skill']) and str(row['EX_Skill']).strip() in requirements.required_ex_skills:
                    required_slots.add(slot)
                    
        return list(required_slots)
    
    def _sort_slots_by_priority(self, slot_equipment: Dict[str, pd.DataFrame], 
                              requirements: EquipmentRequirement) -> List[str]:
        """Sort equipment slots by their potential contribution to meeting requirements."""
        slot_scores = {}
        
        for slot, equipment_df in slot_equipment.items():
            # Calculate maximum potential contribution from this slot
            max_score = 0
            for _, row in equipment_df.iterrows():
                score = self._calculate_contribution_score(row, requirements)
                max_score = max(max_score, score)
            slot_scores[slot] = max_score
            
        # Sort slots by their scores
        return sorted(slot_scores.keys(), key=lambda x: slot_scores[x], reverse=True)

class CombinationWindow:
    def __init__(self, parent, database, equipment_set, on_select_callback):
        self.window = tk.Toplevel(parent)
        self.window.title("Find Equipment Combinations")
        self.window.geometry("800x600")
        self.window.transient(parent)
        self.window.grab_set()
        
        self.database = database
        self.equipment_set = equipment_set
        self.on_select_callback = on_select_callback
        
        # Create optimizer instance
        self.optimizer = EquipmentOptimizer(os.path.join(os.path.dirname(__file__), "equipment.csv"))
        
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create requirements frame
        req_frame = ttk.LabelFrame(main_frame, text="Requirements", padding="5")
        req_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create skill requirements
        skills_frame = ttk.Frame(req_frame)
        skills_frame.pack(fill=tk.X, pady=5)
        
        self.skill_entries = []
        for i in range(3):
            skill_frame = ttk.Frame(skills_frame)
            skill_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(skill_frame, text=f"Skill {i+1}:").grid(row=0, column=0, padx=5)
            skill_var = tk.StringVar(self.window)  # Pass window reference
            skill_combo = ttk.Combobox(skill_frame, textvariable=skill_var, width=30)
            skill_combo['values'] = [''] + sorted(list(database.all_skills))
            skill_combo.grid(row=0, column=1, padx=5)
            
            ttk.Label(skill_frame, text="Min Level:").grid(row=0, column=2, padx=5)
            level_var = tk.StringVar(self.window, value="0")  # Pass window reference
            level_entry = ttk.Entry(skill_frame, textvariable=level_var, width=5, validate="key")
            level_entry['validatecommand'] = (level_entry.register(self.validate_number), '%P')
            level_entry.grid(row=0, column=3, padx=5)
            
            self.skill_entries.append((skill_var, level_var))
        
        # Create stat requirements
        stats_frame = ttk.Frame(req_frame)
        stats_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(stats_frame, text="Min Attack:").grid(row=0, column=0, padx=5)
        self.min_attack_var = tk.StringVar(self.window, value="0")  # Pass window reference
        attack_entry = ttk.Entry(stats_frame, textvariable=self.min_attack_var, width=5, validate="key")
        attack_entry['validatecommand'] = (attack_entry.register(self.validate_number), '%P')
        attack_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(stats_frame, text="Min Defense:").grid(row=0, column=2, padx=5)
        self.min_defense_var = tk.StringVar(self.window, value="0")  # Pass window reference
        defense_entry = ttk.Entry(stats_frame, textvariable=self.min_defense_var, width=5, validate="key")
        defense_entry['validatecommand'] = (defense_entry.register(self.validate_number), '%P')
        defense_entry.grid(row=0, column=3, padx=5)
        
        # Create gender filter
        gender_frame = ttk.Frame(req_frame)
        gender_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(gender_frame, text="Gender:").pack(side=tk.LEFT, padx=5)
        self.gender_var = tk.StringVar(self.window, value="All")  # Pass window reference
        gender_combo = ttk.Combobox(gender_frame, textvariable=self.gender_var, width=10, state='readonly')
        gender_combo['values'] = ['All', 'Male', 'Female']
        gender_combo.pack(side=tk.LEFT, padx=5)
        
        # Create release status filter frame with updated text
        release_frame = ttk.Frame(req_frame)
        release_frame.pack(fill=tk.X, pady=5)
        
        self.include_unreleased_var = tk.BooleanVar(self.window, value=False)
        ttk.Checkbutton(release_frame, text="Include not-released", 
                       variable=self.include_unreleased_var).pack(anchor=tk.W, pady=2)
        
        # Create search options frame
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(options_frame, text="Max Results:").pack(side=tk.LEFT, padx=5)
        self.max_results_var = tk.StringVar(self.window, value="10")  # Pass window reference
        max_results_entry = ttk.Entry(options_frame, textvariable=self.max_results_var, width=5, validate="key")
        max_results_entry['validatecommand'] = (max_results_entry.register(self.validate_number), '%P')
        max_results_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(options_frame, text="Max Checks:").pack(side=tk.LEFT, padx=5)
        self.max_checks_var = tk.StringVar(self.window, value="10000")  # Pass window reference
        max_checks_entry = ttk.Entry(options_frame, textvariable=self.max_checks_var, width=8, validate="key")
        max_checks_entry['validatecommand'] = (max_checks_entry.register(self.validate_number), '%P')
        max_checks_entry.pack(side=tk.LEFT, padx=5)
        
        # Create results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview for results
        self.tree = ttk.Treeview(results_frame, columns=('attack', 'defense', 'skills'), show='headings')
        self.tree.heading('attack', text='Attack')
        self.tree.heading('defense', text='Defense')
        self.tree.heading('skills', text='Skills')
        
        # Configure column widths
        self.tree.column('attack', width=70, anchor='center')
        self.tree.column('defense', width=70, anchor='center')
        self.tree.column('skills', width=600)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        find_button = ttk.Button(button_frame, text="Find Combinations", command=self._find_combinations)
        find_button.pack(side=tk.LEFT, padx=5)
        
        apply_button = ttk.Button(button_frame, text="Apply Selected", command=self._choose_combination)
        apply_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.window.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Store combinations
        self.combinations = []
        
    def validate_number(self, P):
        if P == "":
            return True
        try:
            int(P)
            return True
        except ValueError:
            return False
            
    def _find_combinations(self):
        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get requirements
        required_skills = {}
        for skill_var, level_var in self.skill_entries:
            skill = skill_var.get().strip()
            level = level_var.get().strip()
            if skill and level:
                required_skills[skill] = int(level)
        
        requirements = EquipmentRequirement(
            required_skills=required_skills,
            required_ex_skills=set(),  # TODO: Add EX skill requirements
            min_attack=int(self.min_attack_var.get() or 0),
            min_defense=int(self.min_defense_var.get() or 0),
            gender=self.gender_var.get()
        )
        
        # Get current equipment as selected equipment
        selected_equipment = {}
        for part, equip in self.equipment_set.parts.items():
            if equip is not None:
                selected_equipment[part] = equip.name
        
        # Find combinations
        try:
            # Filter equipment by release status before finding combinations
            if not self.include_unreleased_var.get():
                filtered_equipment = {}
                for part, equips in self.database.equipment_by_part.items():
                    filtered_equipment[part] = [e for e in equips if e.released != "No"]
                self.database.equipment_by_part = filtered_equipment
            
            self.combinations = self.optimizer.find_optimal_combinations(
                requirements=requirements,
                max_results=int(self.max_results_var.get()),
                max_checks=int(self.max_checks_var.get()),
                selected_equipment=selected_equipment
            )
            
            # Display results
            for i, combination in enumerate(self.combinations):
                skills_str = "; ".join([f"{skill}: +{boost}" for skill, boost in combination.skills.items()])
                if combination.ex_skills:
                    skills_str += "; " + "; ".join([f"{skill} (EX): +{boost}" for skill, boost in combination.ex_skills.items()])
                
                self.tree.insert('', 'end', values=(
                    combination.total_attack,
                    combination.total_defense,
                    skills_str
                ))
                
        except Exception as e:
            messagebox.showerror("Error", f"Error finding combinations: {str(e)}")
            
    def _choose_combination(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a combination first.")
            return
            
        selected_idx = self.tree.index(selection[0])
        if 0 <= selected_idx < len(self.combinations):
            self._apply_combination(self.combinations[selected_idx].equipment)
            
    def _apply_combination(self, equipment_dict: Dict[str, str]):
        # Convert equipment names to Equipment objects
        for part, name in equipment_dict.items():
            # Find the equipment in the database
            for equip in self.database.equipment_by_part[part]:
                if equip.name == name:
                    self.equipment_set.parts[part] = equip
                    break
        
        if self.on_select_callback:
            self.on_select_callback()
        
        self.window.destroy()

class EquipmentUI:
    def __init__(self, root: tk.Tk, database: EquipmentDatabase):
        self.database = database
        self.current_set = EquipmentSet()
        self.root = root  # Store the root window reference
        
        # Create frames
        self.equipment_frame = ttk.Frame(self.root)
        self.stats_frame = ttk.Frame(self.root)
        
        self.equipment_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        self._create_equipment_selectors()
        self._create_stats_display()
        
        # Add Find Combinations button in a separate frame below equipment selectors
        button_frame = ttk.Frame(self.equipment_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        find_button = ttk.Button(
            button_frame,
            text="Find Combinations",
            command=self._show_combination_window
        )
        find_button.pack(pady=5)
    
    def _format_equipment_name(self, equipment: Equipment) -> str:
        """Format equipment name with skills for display"""
        if equipment is None:
            return "None"
        
        # Add padding after equipment name (5 spaces)
        parts = [f"{equipment.name}     "]
        
        # Add regular skills
        skill_parts = []
        for skill, boost in equipment.skills.items():
            skill_parts.append(f"{skill}+{boost}")
        if skill_parts:
            parts.append("(" + ", ".join(skill_parts) + ")")  # Close parentheses for regular skills
        
        # Add EX skill if present
        if equipment.ex_skill:
            skill_name, boost = equipment.ex_skill
            # Add EX skill as a separate group
            parts.append(f" (EX: {skill_name} R{boost})")
        
        return "".join(parts)  # Join without spaces since we have padding

    def _create_equipment_selectors(self):
        """Create dropdown menus for each equipment part"""
        self.selectors = {}
        
        # Store name mappings for lookup
        self.name_to_equipment = {}
        
        # Create a frame for all equipment selectors
        selectors_frame = ttk.Frame(self.equipment_frame)
        selectors_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        # Calculate the maximum width of part names for alignment
        max_part_width = max(len(part) for part in self.current_set.parts.keys())
        
        for part in self.current_set.parts.keys():
            frame = ttk.Frame(selectors_frame)
            frame.pack(fill=tk.X, pady=2)
            
            # Create label with fixed width for alignment
            label = ttk.Label(frame, text=part, width=max_part_width + 2)
            label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Create selector frame to hold dropdown and find button
            selector_frame = ttk.Frame(frame)
            selector_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Create and configure the combobox
            var = tk.StringVar()
            dropdown = ttk.Combobox(selector_frame, textvariable=var)
            
            # Create mapping of display names to equipment objects
            self.name_to_equipment[part] = {'None': None}
            equipment_display_names = ['None']
            
            for equip in self.database.equipment_by_part[part]:
                display_name = self._format_equipment_name(equip)
                equipment_display_names.append(display_name)
                self.name_to_equipment[part][display_name] = equip
            
            # Sort alphabetically (keeping None at the top)
            sorted_names = ['None'] + sorted([n for n in equipment_display_names if n != 'None'])
            
            dropdown['values'] = sorted_names
            dropdown.set('None')
            dropdown['state'] = 'readonly'
            dropdown.config(width=60)
            
            # Pack dropdown with expand to take remaining space
            dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Create find button
            find_button = ttk.Button(
                selector_frame, 
                text="Find...",
                command=lambda p=part: self._show_find_window(p)
            )
            find_button.pack(side=tk.LEFT, padx=(5, 0))
            
            # Store references
            self.selectors[part] = {
                'var': var,
                'dropdown': dropdown,
                'values': sorted_names
            }
            
            # Bind events
            var.trace('w', lambda *args, p=part: self._on_selection_change(p))
    
    def _show_find_window(self, part):
        """Show the find window for a specific part"""
        FindWindow(self.root, part, self.database, 
                  lambda name: self.selectors[part]['var'].set(name))

    def _on_selection_change(self, part: str):
        """Handle equipment selection changes"""
        selection = self.selectors[part]['var'].get()
        
        if selection == 'None':
            self.current_set.remove_equipment(part)
        else:
            # Find the selected equipment
            equipment = self.name_to_equipment[part].get(selection)
            if equipment:
                self.current_set.add_equipment(equipment)
        
        self._update_stats_display()
    
    def _update_stats_display(self):
        """Update the displayed stats based on current equipment"""
        totals = self.current_set.calculate_totals()
        
        # Update basic stats
        self.stats_labels['Attack'].config(text=f"Attack: {totals['Attack']}")
        self.stats_labels['Defense'].config(text=f"Defense: {totals['Defense']}")
        
        # Clear existing skill labels
        for widget in self.stats_labels['Skills'].winfo_children():
            widget.destroy()
        for widget in self.stats_labels['EX_Skills'].winfo_children():
            widget.destroy()
        
        # Update skills display
        regular_skills = [(skill, boost) for skill, boost in sorted(totals['Skills'].items()) 
                         if not skill.endswith('(EX)')]
        ex_skills = [(skill.replace(' (EX)', ''), boost) for skill, boost in sorted(totals['Skills'].items()) 
                    if skill.endswith('(EX)')]
        
        # Display regular skills
        for skill, boost in regular_skills:
            skill_label = ttk.Label(self.stats_labels['Skills'], 
                                  text=f"{skill}: +{boost}")
            skill_label.pack(anchor=tk.W, pady=1)
        
        # Display EX skills
        for skill, boost in ex_skills:
            skill_label = ttk.Label(self.stats_labels['EX_Skills'], 
                                  text=f"{skill}: R{boost}")
            skill_label.pack(anchor=tk.W, pady=1)

    def _create_stats_display(self):
        """Create labels to display total stats"""
        # Create a frame for stats with some padding
        stats_container = ttk.Frame(self.stats_frame, padding="10")
        stats_container.pack(fill=tk.X)

        # Basic Stats Section
        basic_stats_frame = ttk.LabelFrame(stats_container, text="Basic Stats", padding="5")
        basic_stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_labels = {}
        
        # Attack and Defense in one row
        basic_row = ttk.Frame(basic_stats_frame)
        basic_row.pack(fill=tk.X)
        
        self.stats_labels['Attack'] = ttk.Label(basic_row, text="Attack: 0")
        self.stats_labels['Attack'].pack(side=tk.LEFT, padx=(0, 20))
        
        self.stats_labels['Defense'] = ttk.Label(basic_row, text="Defense: 0")
        self.stats_labels['Defense'].pack(side=tk.LEFT)

        # Regular Skills Section
        skills_frame = ttk.LabelFrame(stats_container, text="Skill Boosts", padding="5")
        skills_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_labels['Skills'] = ttk.Frame(skills_frame)
        self.stats_labels['Skills'].pack(fill=tk.X)

        # EX Skills Section
        ex_skills_frame = ttk.LabelFrame(stats_container, text="EX Skills", padding="5")
        ex_skills_frame.pack(fill=tk.X)
        
        self.stats_labels['EX_Skills'] = ttk.Frame(ex_skills_frame)
        self.stats_labels['EX_Skills'].pack(fill=tk.X)

    def _show_combination_window(self):
        """Show the combination finder window"""
        CombinationWindow(
            self.root,
            self.database,
            self.current_set,
            self._apply_combination
        )
    
    def _apply_combination(self, equipment_dict: Dict[str, str]):
        """Apply a selected equipment combination"""
        print(f"Applying combination: {equipment_dict}")  # Debug print
        
        # First clear all current selections
        for part in self.current_set.parts:
            self.selectors[part]['var'].set('None')
            self.current_set.remove_equipment(part)
        
        # Then apply the new combination
        for part, name in equipment_dict.items():
            # Find the equipment in the database
            matching_equipment = None
            for equip in self.database.equipment_by_part[part]:
                if equip.name == name:
                    matching_equipment = equip
                    break
            
            if matching_equipment:
                print(f"Found matching equipment for {part}: {name}")  # Debug print
                # Add it to the equipment set
                self.current_set.add_equipment(matching_equipment)
                # Find the display name in the selector values
                display_name = None
                for value in self.selectors[part]['values']:
                    if value != 'None' and value.startswith(name):
                        display_name = value
                        break
                
                if display_name:
                    print(f"Setting {part} to {display_name}")  # Debug print
                    self.selectors[part]['var'].set(display_name)
                else:
                    print(f"Warning: Could not find display name for {name} in {part}")  # Debug print
            else:
                print(f"Warning: Could not find equipment {name} for {part}")  # Debug print
        
        # Update stats display
        self._update_stats_display()

def main():
    # Initialize database
    database = EquipmentDatabase()
    csv_path = get_resource_path("equipment.csv")
    database.load_from_csv(csv_path)
    
    # Create root window
    root = tk.Tk()
    root.title("Equipment Analyzer")
    
    # Create UI
    ui = EquipmentUI(root, database)
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main() 