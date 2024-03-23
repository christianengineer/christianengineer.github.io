import os


def update_markdown_files(folder_path):
    # Get list of markdown files in the specified folder
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]

    # Iterate through each markdown file
    for file_name in markdown_files:
        file_path = os.path.join(folder_path, file_name)
        # Rename the file if it contains multiple dashes
        if "--" in file_name or "---" in file_name or "----" in file_name:
            new_file_name = (
                file_name.replace("--", "-").replace("---", "-").replace("----", "-")
            )
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(file_path, new_file_path)
            print(f"Renamed file: {file_name} -> {new_file_name}")


# Specify the folder path containing markdown files
folder_path = "_posts"

# Call the function to update markdown files in the specified folder
update_markdown_files(folder_path)
