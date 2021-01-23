import Header from '../src/components/Header';
import Nav from '../src/components/Nav';

const LandingPage = () => {
  return (
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
  );
};

export default LandingPage;
