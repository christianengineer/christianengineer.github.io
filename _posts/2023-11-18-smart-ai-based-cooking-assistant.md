---
title: Smart AI-Based Cooking Assistant
date: 2023-11-18
permalink: posts/smart-ai-based-cooking-assistant
layout: article
---

## Smart AI-Based Cooking Assistant - Technical Specifications

## Description

The Smart AI-Based Cooking Assistant is a web-based application that helps users with cooking by providing personalized recipe recommendations, ingredient substitutions, and step-by-step instructions. The application utilizes artificial intelligence techniques to understand user preferences and cooking constraints, allowing for a highly customized cooking experience.

This document focuses on the data management and high user traffic handling capabilities of the Smart AI-Based Cooking Assistant. The objective is to ensure the application can efficiently handle a large volume of user data and scale to support high concurrent user traffic.

## Objectives

1. Efficient Data Management: The application should be able to store and retrieve user data, such as user preferences, favorite recipes, and cooking history, efficiently. The data management system should be scalable and performant.

2. High User Traffic Handling: The application should be able to handle high concurrent user traffic without significant impact on performance. The backend architecture should be designed to scale horizontally.

## Chosen Libraries

### Backend

1. Node.js: Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine. It provides an event-driven, non-blocking I/O model that makes it well-suited for building scalable and high-performance applications. We chose Node.js for its high scalability, rich ecosystem of libraries, and efficient handling of asynchronous operations.

2. Express.js: Express.js is a minimal and flexible Node.js web application framework that provides a robust set of features for web and API development. It allows us to easily handle routing, middleware, and request handling. Express.js is chosen for its simplicity, performance, and compatibility with Node.js.

3. MongoDB: MongoDB is a NoSQL document database that provides high scalability, flexibility, and performance. Its document-oriented model is well-suited for handling complex data structures in a scalable manner. MongoDB also provides features like sharding and replication for horizontal scalability and fault tolerance.

### Frontend

1. React: React is a JavaScript library for building user interfaces. It allows for efficient UI updates through its virtual DOM diffing algorithm and component-based architecture. React's declarative approach to UI development simplifies the codebase and enables better code maintainability and reusability.

2. Redux: Redux is a predictable state container for JavaScript applications, commonly used with React. It provides a centralized state management approach, making it easier to manage application state and data flow. Redux's unidirectional data flow helps in handling complex application states efficiently.

3. Material-UI: Material-UI is a popular React UI framework that follows the Material Design principles. It provides a set of reusable UI components and styles, making it easier to create a visually appealing and responsive user interface. Material-UI is chosen to ensure consistency in design across the application and to reduce development time.

## Tradeoffs

- While MongoDB offers high scalability and performance for document-oriented data storage, it may not be the best choice for applications that require complex relational queries.
- Node.js's event-driven, non-blocking I/O model makes it efficient for handling concurrent requests, but it may not be suitable for CPU-intensive tasks.
- React's virtual DOM and component-based architecture enhance UI performance and code reusability but may have a steeper learning curve for developers new to React.

To facilitate extensive growth and maintain a scalable file structure for the Smart AI-Based Cooking Assistant, we can follow a multi-level directory structure that organizes files based on their functionality and importance. This structure allows for easy navigation, separation of concerns, and modularization. Here is a detailed breakdown of the proposed file structure:

```
├── src
│   ├── server
│   │   ├── config
│   │   │   └── {configuration files}
│   │   ├── controllers
│   │   │   └── {API controller files}
│   │   ├── models
│   │   │   └── {database model files}
│   │   ├── routes
│   │   │   └── {API route files}
│   │   ├── services
│   │   │   └── {business logic files}
│   │   └── app.js
│   ├── client
│   │   ├── components
│   │   │   └── {React component files}
│   │   ├── containers
│   │   │   └── {container component files}
│   │   ├── reducers
│   │   │   └── {Redux reducer files}
│   │   ├── actions
│   │   │   └── {Redux action files}
│   │   ├── styles
│   │   │   └── {CSS or SASS files}
│   │   └── index.js
│   └── shared
│       ├── utils
│       │   └── {utility/helper files}
│       ├── constants
│       │   └── {constant files}
│       └── assets
│           └── {static asset files}
├── public
│   └── index.html
├── .env
├── package.json
└── README.md
```

Let's break down the structure:

- `src`: This directory contains all the source code for the application.

  - `server`: This directory contains server-side files.

    - `config`: Configuration files such as database connection settings and environment variables.
    - `controllers`: API controller files that handle incoming requests and interact with services.
    - `models`: Database model files defining the schema and data structure.
    - `routes`: API route files that define the endpoints and link them to corresponding controllers.
    - `services`: Business logic files that handle complex operations and interact with models.
    - `app.js`: The main entry point for the server application.

  - `client`: This directory contains client-side files, built using React.
    - `components`: React component files that represent reusable UI components.
    - `containers`: Container component files that connect components to Redux and handle data fetching.
    - `reducers`: Redux reducer files that handle state changes for different parts of the application.
    - `actions`: Redux action files that define actions triggered by user or system events.
    - `styles`: CSS or SASS files for styling the UI components.
    - `index.js`: The main entry point for the client-side React application.
  - `shared`: This directory contains shared files used in both server and client applications.
    - `utils`: Utility/helper files that provide common functionalities across the application.
    - `constants`: Constant files that define reusable values used throughout the application.
    - `assets`: Static asset files like images, icons, or fonts used in the application.

- `public`: This directory contains the publicly accessible files for the application.
  - `index.html`: The HTML file used as a template to render the React application on the client-side.

Other files and folders such as `.env` for environment variables, `package.json` for dependencies, and `README.md` for project documentation are kept at the root level.

This file structure provides a clear separation of concerns and facilitates easy collaboration between backend and frontend developers. It also enables adding new features or components without impacting the existing codebase, making it highly scalable as the application grows.

To provide a detailed file for the core logic of the Smart AI-Based Cooking Assistant, let's focus on the backend server's `services` directory where the business logic resides. Here's an example file `recipeService.js` that handles the core functionality related to recipe recommendations and ingredient substitutions:

File path: `src/server/services/recipeService.js`

```javascript
// src/server/services/recipeService.js

const Recipe = require("../models/Recipe");
const UserModel = require("../models/User");

async function getRecommendedRecipes(userId, preferences) {
  try {
    // Fetch user's favorite ingredients from preferences
    const favoriteIngredients = preferences.favoriteIngredients;

    // Fetch user's dietary restrictions from preferences
    const dietaryRestrictions = preferences.dietaryRestrictions;

    // Fetch user's cooking history from the database
    const userCookingHistory = await UserModel.findById(
      userId,
      "cookingHistory",
    ).lean();

    // Perform AI-based algorithm to recommend recipes based on preferences and history
    const recommendedRecipes = await Recipe.find({
      ingredients: { $in: favoriteIngredients },
      dietaryRestrictions: { $nin: dietaryRestrictions },
      _id: { $nin: userCookingHistory.cookingHistory },
    })
      .select("name imageUrl")
      .limit(5)
      .lean();

    return recommendedRecipes;
  } catch (error) {
    // Handle error case
    throw new Error("Failed to get recommended recipes");
  }
}

function substituteIngredients(recipe, ingredientSubstitutions) {
  const substitutedIngredients = recipe.ingredients.map((ingredient) => {
    const substitutedIngredient = ingredientSubstitutions[ingredient.name];
    if (substitutedIngredient) {
      return {
        name: substitutedIngredient.name,
        quantity: ingredient.quantity,
        unit: substitutedIngredient.unit,
      };
    }
    return ingredient;
  });

  return { ...recipe, ingredients: substitutedIngredients };
}

module.exports = {
  getRecommendedRecipes,
  substituteIngredients,
};
```

Explanation:

- The `getRecommendedRecipes` function takes a `userId` and `preferences` as arguments and fetches the user's favorite ingredients, dietary restrictions, and cooking history from the database. It then utilizes an AI-based algorithm to recommend recipes based on the user's preferences and cooking history. The function queries the `Recipe` model to find recipes that contain the user's favorite ingredients, do not violate their dietary restrictions, and have not been previously cooked. It returns the recommended recipes, limited to a maximum of 5.

- The `substituteIngredients` function accepts a `recipe` object and `ingredientSubstitutions` as arguments. It substitutes the ingredients of the given recipe with their corresponding substitutions from the `ingredientSubstitutions` object. It returns the modified recipe with substituted ingredients.

These functions can be imported and used within the controller or routes files where the API endpoints are defined to handle the specific business logic of recommending recipes and ingredient substitutions for the Smart AI-Based Cooking Assistant.

To provide another file for a secondary core logic of the Smart AI-Based Cooking Assistant, let's focus on the backend server's `controllers` directory where the logic for handling API requests resides. Here's an example file `userController.js` that handles user-specific functionality and integration with other files:

File path: `src/server/controllers/userController.js`

```javascript
// src/server/controllers/userController.js

const UserModel = require("../models/User");
const RecipeService = require("../services/recipeService");

async function getUserPreferences(req, res) {
  try {
    const { userId } = req.params;

    // Fetch user preferences from the database
    const user = await UserModel.findById(userId, "preferences").lean();

    res.status(200).json(user.preferences);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch user preferences" });
  }
}

async function getRecommendedRecipes(req, res) {
  try {
    const { userId } = req.params;

    // Fetch user preferences from the database
    const user = await UserModel.findById(userId, "preferences").lean();

    // Get recommended recipes matching user preferences
    const recommendedRecipes = await RecipeService.getRecommendedRecipes(
      userId,
      user.preferences,
    );

    res.status(200).json(recommendedRecipes);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch recommended recipes" });
  }
}

module.exports = {
  getUserPreferences,
  getRecommendedRecipes,
};
```

Explanation:

- The `getUserPreferences` function is responsible for handling requests to fetch the user's preferences from the database. It receives the `userId` from the request parameters, queries the `UserModel`, and responds with the user's preferences in the response.

- The `getRecommendedRecipes` function handles requests to retrieve recommended recipes for the user based on their preferences. It utilizes the `RecipeService` module we defined earlier to fetch the user's preferences from the database and call the `getRecommendedRecipes` function, passing the `userId` and user's preferences. The recommended recipes are then returned in the response.

These controller functions can be mapped to the corresponding API routes and integrated with other parts of the application. For example, in a separate file within the `routes` directory, we can define routes that link to these controller functions:

File path: `src/server/routes/userRoutes.js`

```javascript
// src/server/routes/userRoutes.js

const express = require("express");
const userController = require("../controllers/userController");

const router = express.Router();

router.get("/:userId/preferences", userController.getUserPreferences);
router.get(
  "/:userId/recommended-recipes",
  userController.getRecommendedRecipes,
);

module.exports = router;
```

In this example, we define two routes: one to fetch user preferences (`/userId/preferences`) and another to get recommended recipes (`/userId/recommended-recipes`). These routes are linked to the corresponding controller functions defined in `userController.js`.

By organizing the logic into separate controllers, we can maintain a modular and scalable codebase. The `userController.js` file integrates with the `UserModel` and `RecipeService` files to handle user-specific functionality and respond to API requests related to user preferences and recommended recipes.

To outline an additional core logic for the Smart AI-Based Cooking Assistant, let's focus on the backend server's `models` directory where the data models are defined. Here's an example file `Recipe.js` that represents the recipe model and its interdependencies with other files:

File path: `src/server/models/Recipe.js`

```javascript
// src/server/models/Recipe.js

const mongoose = require("mongoose");

const recipeSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  ingredients: [
    {
      name: {
        type: String,
        required: true,
      },
      quantity: {
        type: Number,
        required: true,
      },
      unit: {
        type: String,
        required: true,
      },
    },
  ],
  imageUrl: {
    type: String,
    required: true,
  },
  dietaryRestrictions: [
    {
      type: String,
      enum: ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free"],
    },
  ],
});

module.exports = mongoose.model("Recipe", recipeSchema);
```

Explanation:

- The `Recipe.js` file defines a Mongoose schema for the `Recipe` model, which represents a recipe. It includes fields such as `name`, `ingredients`, `imageUrl`, and `dietaryRestrictions`.
- The `name` field represents the name of the recipe and is required.
- The `ingredients` field is an array of objects, where each object represents an ingredient with properties `name`, `quantity`, and `unit`. Each ingredient object is required.
- The `imageUrl` field specifies the URL of the image associated with the recipe and is required.
- The `dietaryRestrictions` field is an array of strings representing the dietary restrictions that apply to the recipe. It uses the `enum` option to restrict the possible values to predefined options.

This model file plays a crucial role within the system as it represents the structure of the recipe data and defines the schema for storing and retrieving recipe information in the MongoDB database.

The `Recipe` model can be imported and used in several other files, such as the `recipeService.js` file we previously discussed. In the `recipeService.js` file, the `Recipe` model is imported to query the database and perform operations on recipe data.

Similarly, the controller file (`userController.js`) that handles API requests related to recipes may also require importing the `Recipe` model to interact with recipe data, such as retrieving recommended recipes or performing ingredient substitutions.

By defining the `Recipe` model in a separate file, we create a modular and scalable structure that allows separation of concerns, easy data manipulation, and seamless integration with other parts of the Smart AI-Based Cooking Assistant system.

Here are different types of users who will use the Smart AI-Based Cooking Assistant application along with their user stories and the files that would accomplish them:

1. Home Cook (Regular User):
   User Story: As a home cook, I want to explore a variety of recipes based on my preferences and dietary restrictions, so I can prepare delicious meals for myself and my family.
   File: `userController.js` (in `src/server/controllers` directory) would handle the API request to fetch recommended recipes based on user preferences and dietary restrictions.

2. Novice Cook:
   User Story: As a novice cook, I want step-by-step instructions and cooking tips for each recipe, so I can learn and improve my cooking skills.
   File: `recipeService.js` (in `src/server/services` directory) would contain the logic to retrieve and provide step-by-step cooking instructions along with relevant cooking tips for each recipe.

3. Ingredient Allergy Prone User:
   User Story: As an ingredient allergy-prone user, I need to be able to easily substitute ingredients in recipes, so I can avoid allergic reactions while still enjoying delicious meals.
   File: `recipeService.js` (in `src/server/services` directory) would include the logic to handle ingredient substitutions within recipes, allowing users to substitute problematic ingredients with suitable alternatives.

4. Health-Conscious User:
   User Story: As a health-conscious user, I want personalized recipe recommendations that align with my dietary goals, so I can maintain a healthy lifestyle.
   File: `userController.js` (in `src/server/controllers` directory) would handle the API request to fetch recommended recipes based on user preferences that are specifically tailored to their dietary goals and health requirements.

5. Recipe Contributor:
   User Story: As a recipe contributor, I want to be able to add, edit, and delete my own recipes, so I can share my favorite dishes with the community.
   File: `recipeController.js` (in `src/server/controllers` directory) would handle the API requests for the creation, modification, and deletion of recipes. This file would integrate with the `Recipe.js` model (in `src/server/models` directory) for proper data storage and retrieval.

These user types and their corresponding user stories can be accommodated by different parts of the application, such as the backend controllers (`userController.js`, `recipeController.js`), services (`recipeService.js`), and
