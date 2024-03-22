import os
import re
import yaml


def update_permalink_dashes(folder_path):
    # Get list of markdown files in the specified folder
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]

    # Iterate through each markdown file
    for file_name in markdown_files:
        file_path = os.path.join(folder_path, file_name)
        # Read the content of the file
        with open(file_path, "r") as file:
            content = file.read()

        # Find front matter
        front_matter_pattern = r"^---\n(.*?\n)---\n"
        match = re.search(front_matter_pattern, content, re.MULTILINE | re.DOTALL)

        if match:
            front_matter = match.group(1)
            # Parse front matter as YAML
            front_matter_data = yaml.safe_load(front_matter)
            if "permalink" in front_matter_data:
                permalink = front_matter_data["permalink"]
                # Replace multiple dashes with a single dash in the permalink
                updated_permalink = re.sub(r"-{2,}", "-", permalink)
                # Replace old permalink with updated one in the content
                updated_content = content.replace(permalink, updated_permalink)

                # Write the modified content back to the file
                with open(file_path, "w") as file:
                    file.write(updated_content)


# Specify the folder path containing markdown files
folder_path = "_posts"

# Call the function to update permalinks in markdown files in the specified folder
update_permalink_dashes(folder_path)
