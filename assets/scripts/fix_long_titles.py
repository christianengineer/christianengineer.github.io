import os
import re
import yaml
from openai import OpenAI
import random
import time

# Remove comments to test locally
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Set up your OpenAI API key here


def add_pause(min_seconds, max_seconds):
    # Generate a random duration between min_seconds and max_seconds
    duration = random.uniform(min_seconds, max_seconds)
    # Pause execution for the specified duration
    time.sleep(duration)


def remove_special_characters(text):
    # Define regex pattern to match special characters
    pattern = r"[^a-zA-Z0-9\s]"
    # Remove special characters using regex substitution
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def generate_seo_title_and_description(title):
    this_tokens = 0

    title = remove_special_characters(title)
    # Define the prompt to generate SEO-optimized title
    title_prompt = f"Generate a more technical title from, {title}. Use this format: [shocking project problem], [AI tool or library] for [project soluction]. Response should be a max of 65 characters"
    # title_prompt = f"Analyze {title}, then generate a new title that succinctly describes using AI to solve a specific problem. Response should be a max of 65 characters"
    conversation_title = [
        {"role": "user", "content": title_prompt},
    ]

    # Call OpenAI to generate SEO-optimized title
    title_response = client.chat.completions.create(
        # expensive
        # model="gpt-4-1106-preview",
        # cheap
        # model="gpt-3.5-turbo-1106",
        # cheaper, better
        model="gpt-3.5-turbo-0125",
        messages=conversation_title,
    )

    generated_title = title_response.choices[0].message.content
    this_tokens += title_response.usage.total_tokens

    # Define the prompt to generate SEO-optimized description
    description_prompt = f"For the project: {generated_title}, what are the AI tools and libraries we'll be using and why?. Response should be between 145 to 155 characters"

    conversation_description = [
        {"role": "user", "content": description_prompt},
    ]

    # Call OpenAI to generate SEO-optimized description
    description_response = client.chat.completions.create(
        # expensive
        # model="gpt-4-1106-preview",
        # cheap
        # model="gpt-3.5-turbo-1106",
        # cheaper, better
        model="gpt-3.5-turbo-0125",
        messages=conversation_description,
    )
    generated_description = description_response.choices[0].message.content
    this_tokens += description_response.usage.total_tokens

    return generated_title, generated_description, this_tokens


def update_markdown_files(folder_path):
    total_tokens = 0
    # Get list of markdown files in the specified folder
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]

    # Iterate through each markdown file
    for file_name in markdown_files:
        tokens_so_far = 0
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
            if "title" in front_matter_data:
                # add pause
                add_pause(3, 5)
                # Generate SEO-optimized title and description
                generated_title, generated_description, this_tokens = (
                    generate_seo_title_and_description(front_matter_data["title"])
                )

                # Add generated title and description to the front matter
                front_matter_data["title"] = generated_title
                front_matter_data["description"] = generated_description
                total_tokens += this_tokens
                tokens_so_far += this_tokens

                # Convert front matter back to YAML format
                updated_front_matter = yaml.dump(
                    front_matter_data,
                    default_flow_style=False,
                    default_style=None,
                    width=float("inf"),
                )

                # Replace old front matter with updated one in the content
                updated_content = content.replace(front_matter, updated_front_matter)
                print(f"\n\n\nupdated_front_matter: {updated_front_matter}\n\n\n\n")

                # Write the modified content back to the file
                with open(file_path, "w") as file:
                    file.write(updated_content)
        print(f"tokens_so_far: {tokens_so_far}")
    print(f"total_tokens: {total_tokens}")
    print(f"tokens_so_far: {tokens_so_far}")


# Specify the folder path containing markdown files
folder_path = "_posts"

# Call the function to update markdown files in the specified folder
update_markdown_files(folder_path)
