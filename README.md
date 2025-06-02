# GVO Equipment Analyzer

A tool for analyzing and finding optimal equipment combinations in GVO.

## Features

- View and filter equipment by skills, gender, and release status
- Find optimal equipment combinations based on requirements
- Calculate total stats and skill boosts
- Support for regular skills and EX skills
- Filter equipment by release status (Global/Japan)

## Installation

1. Make sure you have Python 3.7 or newer installed. You can download it from [python.org](https://www.python.org/downloads/).

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure your equipment data is in the `equipment.csv` file with the following columns:
   - Part
   - Name
   - Att
   - Def
   - Skill_1, Boost_1
   - Skill_2, Boost_2
   - Skill_3, Boost_3
   - EX_Skill, Boost_Ex
   - Male/Female/All
   - Released (Yes/Japan)

2. Run the analyzer:
   ```bash
   python equipment_analyzer.py
   ```

3. Using the interface:
   - Select equipment using the dropdown menus or "Find..." buttons
   - Use "Find Combinations" to search for optimal equipment sets
   - Filter equipment by skills, gender, and release status
   - View total stats and skill boosts in real-time

## Files Included

- `equipment_analyzer.py` - The main program
- `equipment.csv` - Your equipment database
- `requirements.txt` - Required Python packages
- `README.md` - This documentation

## Notes

- The "Include not-released" option shows equipment that's only available in the Japanese version
- Equipment combinations are sorted by their effectiveness in meeting requirements
- The tool automatically calculates the best combinations based on your criteria

## Support

If you encounter any issues or have questions, please report them on the project's issue tracker. 