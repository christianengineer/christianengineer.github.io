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
          backgroundGradient="education"
          align="right"
          sectionTheme="dark"
          linkTheme="dark"
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
      </main>
    </>
  );
};

export default LandingPage;
