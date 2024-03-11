import os
import re


def update_markdown_files(folder_path):
    # Get list of markdown files in the specified folder
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]

    # Iterate through each markdown file
    for file_name in markdown_files:
        file_path = os.path.join(folder_path, file_name)
        # Read the content of the file
        with open(file_path, "r") as file:
            content = file.read()

        # Perform the replacements using regular expression
        updated_content = re.sub(r"(#+) \*\*(.*?)\*\*", r"\1 \2", content)

        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.write(updated_content)


# Specify the folder path containing markdown files
folder_path = "_posts"

# Call the function to update markdown files in the specified folder
update_markdown_files(folder_path)
