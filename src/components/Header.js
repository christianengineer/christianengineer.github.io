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

export default Header;
