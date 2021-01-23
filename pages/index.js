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
      </main>
    </>
  );
};

export default LandingPage;
