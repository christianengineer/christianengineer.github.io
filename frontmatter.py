import os

def update_markdown_files(folder_path):
    # Get list of markdown files in the specified folder
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]

    # Iterate through each markdown file
    for file_name in markdown_files:
        file_path = os.path.join(folder_path, file_name)
        # Read the content of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Check if the first line contains the front matter delimiter "---"
        if len(lines) > 0 and lines[0].strip() == '---':
            # Find the index of the second occurrence of "---"
            second_delimiter_index = find_second_delimiter_index(lines)

            # Insert 'layout: article' after the first line with "---"
            if second_delimiter_index is not None:
                lines.insert(second_delimiter_index, 'layout: article\n')

            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)

def find_second_delimiter_index(lines):
    # Iterate through the lines to find the second occurrence of "---"
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == '---':
            return i
    return None

# Specify the folder path containing markdown files
folder_path = '_posts'

# Call the function to update markdown files in the specified folder
update_markdown_files(folder_path)
