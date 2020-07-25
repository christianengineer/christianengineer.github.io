# Reading unfamiliar code

Source: https://www.youtube.com/watch?v=wN4ZuGruiNw

Use of context

- How does it relate to what I know about the application and its usage.
- Make connections between what I am currently reading vs what I experienced in reading other code.

Embrace confusion

- Instead of understanding the entire codebase, start with first understanding one particular feature. Ignore anything that is unrelated to that particular feature.

Tools

- Log to write down what I am looking at, at any given point.
- Log what you are currently thinking.
- File names, directory names
- Do as much as possible so you can be as efficient as possible in understanding what you are doing (investigating) and reading.
- Aggressively ignore things, but write them down if they are worth coming back to.
- Focused on the code that might be the most useful.

Start

- Start with elements of the user interface, which are unique identifiers that I can search for in the code. Such as a string identifer on the interface.
- Find a point where you can find out more about that particular feature.
- Go through files / directories in triage for that particular feature, which results in focused understanding.

Repeat with another identifer / feature. (priority of something that sounds useful)

Do not read top to bottom.
Do not go from main function to small functions.

Other Tools

- A list of defined functions
- A list of all defined classes.
- Tools that will minimize the amount of information shown to me so I can make fewer decisions.
- Code folding +/-
- History of the file or commit history.
- Check for multiple files changed in a commit, it presumably addresses a problem that is described in the commit message. Focus to see what changes were made to solve a problem in the past. Can be use as context to help me understand other parts of the file that I can read. Useful for deciding that is most important to start reading first.

Iterative process

- Keep priority queue of things you want to look for in the future.
- List of interesting items.
- List of things I do not understand, their purpose or involvement in the project.
- Do not get sidetrack by things you do not understand. (Rabbit hole avoidance).
- Embrace confusion. Tame confusion.
- Once finish, move on to the next.
- Breadth first search, rather than depth first search.

Identifying relationships

- How the current code you are reading interacts with other code.
- Lean on compiler.
- Small refactoring. Encapsulating code if its being repeated. Making functions private to see who calls it.
- Identify points of interaction between code. Other classes inheriting from the code you are reading. Other code instantiating the code you are reading. Importing. Including. To find out how the code is being used to provide a broader context.
- Use a debugger to understanding the code in the runtime of the program. Insert a breakpoint to see where the program execution is interrupted and also see what other code is calling the code are you are reading, understanding the context around it.
- Print / Console Log statements to see how often, and when does this code get called.
- Identify things that are similarly named but slightly different.
- Identify design patterns. (Model, Viewer, Controller named folders)
- Decide what to investigate next based on how relevant it is to the previous code.

Avoid long files that do not contain useful information.

When finished and you feel you don't understand enough. Start over again but with different search keywords. Look for other parts of the user interface that might be relevant to what you are trying to accomplish.

Summary

- List of things I found confusing.
- List of things I understand.
- List of assumptions I have made.
- List of guesses that might be reasonable.
- Gives use far more to work with and we have now seen patterns that have emerged in the code we read. It gives me more clarity of what I could ignore because is common everywhere and avoid unnecessary investigating.
