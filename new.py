import re

# Read the current requirements file
with open('requirements.txt', 'r') as file:
    lines = file.readlines()

# Open a new file to write the updated requirements
with open('requirements_no_version.txt', 'w') as file:
    for line in lines:
        # Remove version constraint
        new_line = re.sub(r'==.*', '', line).strip()
        if new_line:  # Ensure the line is not empty
            file.write(f'{new_line}\n')
