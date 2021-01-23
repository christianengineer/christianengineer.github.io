import styled from 'styled-components';
import Header from '../src/components/Header';
import Nav from '../src/components/Nav';

const ContentLeft = styled.div`
  background-color: ${({ theme }) => theme.colors.accentColor};
  display: flex;
  align-content: center;
  justify-content: center;
  flex-direction: column;
  width: 50vw;

  @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
    animation: 1.2s ease-out 0s 1 slideInFromLeft;
    width: 100%;
  }

  & h1,
  h2 {
    letter-spacing: -2px;
    line-height: 0.886;

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneSmall}) {
      letter-spacing: -6px;
    }
  }

  & h1 {
    transform: rotate(-90deg);
    font-size: 5rem;
    margin-top: -280px;
    margin-left: -90px;
    animation: 1.4s ease-out 0s 1 slideInFromBottom;

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneSmall}) {
      font-size: 8rem;
      margin-top: -270px;
      margin-left: -120px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
      animation: 1.4s ease-out 0s 1 slideInFromRight;
      margin: 0;
      transform: none;
      padding-left: 2%;
      font-size: 17vw;
      display: block;
    }
  }

  & h2 {
    color: ${({ theme }) => theme.colors.accentSecondaryColor};
    transform: rotate(-90deg);
    font-weight: 300;
    font-size: 4rem;
    margin-top: -72px;
    margin-left: 0px;
    animation: 1.4s ease-out 0s 1 slideInFromTop;

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneSmall}) {
      font-size: 5.57rem;
      margin-top: -110px;
      margin-left: 0px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
      transform: none;
      margin: 0;
      animation: 1s ease-out 0s 1 slideInFromLeft;
      font-size: 10.26vw;
      padding-left: 35.2%;
      display: block;
    }
  }
`;

const ContentRight = styled.div`
  background-color: ${({ theme }) => theme.colors.primaryColor};
  width: 50vw;

  @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
    width: 28vw;
  }
`;

const NavLink = styled.a`
  color: inherit;
  text-decoration: none;
  padding: 15px;
  border-bottom: 1px solid transparent;
  border-left: 1px solid transparent;
  transition: 0.5s ease-in-out;

  &:hover {
    border-bottom: 1px solid ${({ theme }) => theme.colors.primaryColor};
    border-left: 1px solid ${({ theme }) => theme.colors.primaryColor};
  }
`;

const LandingPage = () => {
  return (
    <Header>
      <Nav top>
        <div>
          <NavLink href="https://github.com/chrisipanaque" target="_blank">
            Github
          </NavLink>
        </div>
        <div>
          <NavLink
            href="https://www.linkedin.com/in/chrisipanaque/"
            target="_blank"
          >
            LinkedIn
          </NavLink>
        </div>
        <div>
          <NavLink href="./christian_ipanaque_2019.pdf" target="_blank">
            Resume
          </NavLink>
        </div>
      </Nav>
      <ContentLeft>
        <h1>Christian</h1>
        <h2>Ipanaque</h2>
      </ContentLeft>
      <ContentRight />
      <Nav bottom>
        <div>
          <NavLink href="#education">Education</NavLink>
        </div>
        <div>
          <NavLink href="#achievements">Achievements</NavLink>
        </div>
        <div>
          <NavLink href="#projects">Projects</NavLink>
        </div>
      </Nav>
    </Header>
  );
};

export default LandingPage;
