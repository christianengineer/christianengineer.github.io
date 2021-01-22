import styled from 'styled-components';

const Header = styled.div`
  color: ${({ theme }) => theme.colors.primaryColor};
  position: relative;
  display: flex;
  justify-content: space-between;
  text-transform: uppercase;
  height: 410px;

  @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
    height: 400px;
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.phoneLarge}) {
    height: 450px;
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.iPadMedium}) {
    height: 500px;
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.iPadLarge}) {
    height: 550px;
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.smallLaptop}) {
    height: 600px;
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.largeLaptop}) {
    height: 666px;
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.smallDesktop}) {
    height: 789px;
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.mediumDesktop}) {
    height: 939px;
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.largeDesktop}) {
    height: 969px;
  }

  nav div a {
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
  }

  &::after {
    position: absolute;
    content: '';
    animation: 1.4s ease-out 0s 1 slideInFromRight;
    background-size: 350px;
    background-image: url(./images/christian-header-background-medium.png);
    background-position: center;
    background-repeat: no-repeat;
    width: 100%;
    height: 410px;
    margin-left: auto;
    margin-right: auto;

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
      animation: 1s ease-out 0s 1 slideInFromRight;
      background-size: cover;
      background-image: url(./images/christian-header-background-large.png);
      right: 0;
      width: 330px;
      height: 400px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneLarge}) {
      width: 45vw;
      height: 450px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.iPadMedium}) {
      height: 500px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.iPadLarge}) {
      height: 550px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.smallLaptop}) {
      height: 600px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.largeLaptop}) {
      height: 666px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.smallDesktop}) {
      height: 789px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.mediumDesktop}) {
      height: 939px;
    }

    @media (min-width: ${({ theme }) => theme.breakPoints.largeDesktop}) {
      height: 969px;
    }
  }
`;

const NavBottom = styled.nav`
  display: none;
  margin-bottom: 40px;
  margin-left: 45px;
  position: absolute;
  justify-content: center;
  opacity: 0;
  font-size: 1.6rem;
  animation: 1.2s ease-out 2s forwards fadeIn;
  bottom: 0;
  z-index: 1;

  div {
    padding-right: 10px;

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneLarge}) {
      padding-right: 50px;
    }
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
    display: flex;
  }
`;

const NavTop = styled.nav`
  display: none;
  margin-top: 40px;
  position: absolute;
  justify-content: center;
  opacity: 0;
  font-size: 1.6rem;
  animation: 1.2s ease-out 1.5s forwards fadeIn;
  right: 30%;
  z-index: 1;

  div {
    padding-right: 10px;

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneLarge}) {
      padding-right: 50px;
    }
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
    display: flex;
  }
`;

const HeaderLeft = styled.div`
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

const HeaderRight = styled.div`
  background-color: ${({ theme }) => theme.colors.primaryColor};
  width: 50vw;

  @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
    width: 28vw;
  }
`;

const LandingPage = () => {
  return (
    <Header>
      <NavTop>
        <div>
          <a href="https://github.com/chrisipanaque" target="_blank">
            Github
          </a>
        </div>
        <div>
          <a href="https://www.linkedin.com/in/chrisipanaque/" target="_blank">
            LinkedIn
          </a>
        </div>
        <div>
          <a href="./christian_ipanaque_2019.pdf" target="_blank">
            Resume
          </a>
        </div>
      </NavTop>
      <HeaderLeft>
        <h1>Christian</h1>
        <h2>Ipanaque</h2>
      </HeaderLeft>
      <HeaderRight />
      <NavBottom>
        <div>
          <a href="#education">Education</a>
        </div>
        <div>
          <a href="#achievements">Achievements</a>
        </div>
        <div>
          <a href="#projects">Projects</a>
        </div>
      </NavBottom>
    </Header>
  );
};

export default LandingPage;
