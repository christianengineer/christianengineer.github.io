import Header from '../src/components/Header';
import Nav from '../src/components/Nav';
import Section from '../src/components/Section';

const LandingPage = () => {
  return (
    <>
      <Header>
        <Nav top>
          <div>
            <Nav.Link href="https://github.com/chrisipanaque" target="_blank">
              Github
            </Nav.Link>
          </div>
          <div>
            <Nav.Link
              href="https://www.linkedin.com/in/chrisipanaque/"
              target="_blank"
            >
              LinkedIn
            </Nav.Link>
          </div>
          <div>
            <Nav.Link href="./christian_ipanaque_2019.pdf" target="_blank">
              Resume
            </Nav.Link>
          </div>
        </Nav>
        <Header.Content left>
          <h1>Christian</h1>
          <h2>Ipanaque</h2>
        </Header.Content>
        <Header.Content right />
        <Nav bottom>
          <div>
            <Nav.Link href="#education">Education</Nav.Link>
          </div>
          <div>
            <Nav.Link href="#achievements">Achievements</Nav.Link>
          </div>
          <div>
            <Nav.Link href="#projects">Projects</Nav.Link>
          </div>
        </Nav>
      </Header>

      <main>
        <Section
          id="education"
          backgroundGradient="education"
          align="right"
          sectionTheme="dark"
        >
          <h2>Education</h2>
          <div>
            <h3>Lambda School</h3>
            <h4>
              <a href="https://lambdaschool.com/curriculum" target="_blank">
                View Curriculum
              </a>
            </h4>
            <h4>
              9-month Full Stack Software Development and Computer Science
              Bootcamp that provides a full-time immersive hands-on training
              curriculum. Received training in cross-functional team
              collaboration, and constantly going above and beyond in meeting
              code standards.
            </h4>
          </div>
        </Section>

        <Section
          id="achievements"
          backgroundGradient="achievements"
          align="left"
          sectionTheme="light"
        >
          <h2>Achievements</h2>
          <div>
            <h3>AWS Certified Solutions Architect</h3>
            <h4>
              <a
                href="https://www.youracclaim.com/badges/ea7d8a27-1a69-466b-9e80-68803c43d8d5/public_url"
                target="_blank"
              >
                View Certificate
              </a>
            </h4>
            <h4>
              Designed cost-efficient and scalable systems using Amazon Web
              Services.
            </h4>
          </div>
          <div>
            <h3>AWS Certified Developer</h3>
            <h4>
              <a
                href="https://www.youracclaim.com/badges/ab67a10d-7bd5-4d28-a632-a2b332e5ef45/public_url"
                target="_blank"
              >
                View Certificate
              </a>
            </h4>
            <h4>
              Developed, deployed and debugged cloud-based applications using
              Amazon Web Services.
            </h4>
          </div>
        </Section>

        <Section
          id="projects"
          backgroundGradient="projects"
          align="left"
          sectionTheme="dark"
        >
          <h2>Projects</h2>
          <div>
            <h3>GraphQL Blog</h3>
            <h4>
              <a
                href="https://graphql-apollo-prisma-blog.herokuapp.com/"
                target="_blank"
              >
                Live Demo
              </a>
              |
              <a
                href="https://github.com/chrisipanaque/graphql-prisma-postgresql-blog/"
                target="_blank"
              >
                Github Repository
              </a>
            </h4>
            <h4>
              • Built blog app using the latest Apollo Client Hooks, useQuery
              and useMutation.
            </h4>
            <h4>
              • Improved building speed with less boilerplate code by using
              Apollo Server.
            </h4>
            <h4>
              • Simplified database access by using Prisma Client type safe data
              layer that replaces traditional ORMs.
            </h4>
            <h4>
              <span>Tools:</span> GraphQL, Apollo Client, Apollo Server, Prisma,
              PostgreSQL, React Hooks, MaterialUI.
            </h4>
          </div>
          <div>
            <h3>GraphQL Todo List</h3>
            <h4>
              <a
                href="https://graphql-mongodb-todo-list.herokuapp.com/"
                target="_blank"
              >
                Live Demo
              </a>
              |
              <a
                href="https://github.com/chrisipanaque/graphql-mongodb-todo-list/"
                target="_blank"
              >
                Github Repository
              </a>
            </h4>
            <h4>
              • Rebuilt to-do list app using the latest Apollo Client Hooks,
              useQuery and useMutation.
            </h4>
            <h4>
              • Optimized database insertion and retrieval speeds by replacing
              SQL database with MongoDB.
            </h4>
            <h4>
              <span>Tools:</span> GraphQL, Express, Mongoose, MongoDB, Apollo
              Client, React Hooks, MaterialUI.
            </h4>
          </div>
          <div>
            <h3>Startup Success Predictor</h3>
            <h4>
              <a href="https://predict-a-venture.netlify.com/" target="_blank">
                Live Demo
              </a>
              |
              <a href="https://github.com/bw-startup/frontend/" target="_blank">
                Github Repository
              </a>
            </h4>
            <h4>
              • Led front end development on a cross-functional team of 5
              (composed of 2 Data Scientists and 3 Web Developers), using data
              driven development prior to project implementation.
            </h4>
            <h4>
              • Optimized state machine design pattern by using React Hooks for
              state management, replacing Redux.
            </h4>
            <h4>
              • Improved user experience by allowing user to save past
              predictions using the React Hook; useCookies.
            </h4>
            <h4>
              • Achieved a pleasant UI that matched our audience using Styled
              Components.
            </h4>
            <h4>
              <span>Tools:</span> React Hooks; useState, useEffect, useContext,
              useReducer, useCookies. Styled Components.
            </h4>
          </div>
        </Section>

        <Section
          backgroundGradient="publications"
          align="right"
          sectionTheme="light"
        >
          <h2>Publications</h2>
          <div>
            <h3>
              The 21 sentences I read every day to keep myself mentally strong.
            </h3>
            <h4>
              <a
                href="https://www.linkedin.com/pulse/21-sentences-i-read-every-day-keep-myself-mentally-strong-ipanaque/"
                target="_blank"
              >
                View Article
              </a>
            </h4>
            <h4>
              Article written to demonstrate that mental strength, as any other
              skill, can be developed with practice. The daily practice of
              pushing yourself to grow stronger, maintaining realistic optimism,
              and setting healthy boundaries.
            </h4>
          </div>
        </Section>

        <Section backgroundGradient="ethics" align="left" sectionTheme="dark">
          <h2>Ethics and Values</h2>
          <div>
            <h3>About Me</h3>
            <h4>
              I am a Full Stack Software Development Engineer living in Seattle,
              WA who is passionate about scalable software design patterns and
              development paradigms that provide solutions to local businesses.
            </h4>
          </div>
          <div>
            <h3>Work Ethic</h3>
            <h4>
              I am an innovative and strategic thinking professional with a
              proven track record of consistently going above and beyond in
              meeting customer needs, dedicated to maintaining a reputation
              built on quality, service and uncompromising ethics.
            </h4>
          </div>
        </Section>
      </main>
    </>
  );
};

export default LandingPage;
