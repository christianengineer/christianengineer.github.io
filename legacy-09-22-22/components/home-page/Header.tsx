import styled, { css } from 'styled-components';
import { HeaderAnimations } from '@styles';

type HeaderContentType = {
  left?: boolean;
  right?: boolean;
};

export const Header = styled.div`
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

  @media (min-width: ${({ theme }) => theme.breakPoints.extraLargeDesktop}) {
    height: 1090px;
  }

  @media (min-width: ${({ theme }) =>
      theme.breakPoints.superExtraLargeDesktop}) {
    height: 1290px;
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

    @media (min-width: ${({ theme }) => theme.breakPoints.extraLargeDesktop}) {
      height: 1090px;
    }

    @media (min-width: ${({ theme }) =>
        theme.breakPoints.superExtraLargeDesktop}) {
      height: 1290px;
    }
  }

  ${HeaderAnimations}
`;

export const HeaderContent = styled.div<HeaderContentType>`
  ${({ left }) =>
    left &&
    css`
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
    `}

  ${({ right }) =>
    right &&
    css`
      background-color: ${({ theme }) => theme.colors.primaryColor};
      width: 50vw;

      @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
        width: 28vw;
      }
    `}
`;
