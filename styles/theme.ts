import { DefaultTheme, css } from 'styled-components';

// new colors:
// blue #B6C4E4
// pink #E69FC0
// light pink #F8D1DA

export const theme: DefaultTheme = {
  colors: {
    primaryColor: '#3a3729',
    secondaryColor: '#fbfbfb',
    lightGrayColor: '#D8D8D8',
    accentColor: '#fff95a',
    accentSecondaryColor: '#898632',
  },
  gradients: {
    light: `linear-gradient(rgba(255, 249, 90, 0.8), rgba(255, 249, 90, 0.8))`,
    dark: `linear-gradient(rgba(58, 55, 41, 0.8), rgba(58, 55, 41, 0.8))`,
  },
  fonts: {
    primaryFont: `'Oswald', 'sans-serif'`,
  },
  breakPoints: {
    phoneSmall: '400px',
    phoneMedium: '660px',
    phoneLarge: '760px',

    iPadMedium: '880px',
    iPadLarge: '980px',

    smallLaptop: '1080px',
    largeLaptop: '1200px',

    smallDesktop: '1400px',
    mediumDesktop: '1600px',
    largeDesktop: '1900px',
    extraLargeDesktop: '2000px',
    superExtraLargeDesktop: '2300px',
  },
  sectionBackgrounds: {
    education: css`
      background: ${({ theme }) => theme.gradients.dark},
        url(./images/christian-ipanaque-education.jpg);
    `,
    achievements: css`
      background: ${({ theme }) => theme.gradients.light},
        url(./images/christian-ipanaque-achievements.jpg);
    `,
    projects: css`
      background: ${({ theme }) => theme.gradients.dark},
        url(./images/christian-ipanaque-projects.jpg);
    `,
    publications: css`
      background: ${({ theme }) => theme.gradients.light},
        url(./images/christian-ipanaque-publications.jpg);
    `,
    ethics: css`
      background: ${({ theme }) => theme.gradients.dark},
        url(./images/christian-ipanaque-ethics.jpg);
    `,
  },
  sectionThemes: {
    light: css`
      h2 {
        color: ${({ theme }) => theme.colors.primaryColor};
      }

      h3 {
        color: ${({ theme }) => theme.colors.accentSecondaryColor};
      }

      h4 {
        color: ${({ theme }) => theme.colors.accentSecondaryColor};
      }

      a {
        color: ${({ theme }) => theme.colors.primaryColor};
      }
    `,
    dark: css`
      h2 {
        color: ${({ theme }) => theme.colors.accentColor};
      }

      h3 {
        color: ${({ theme }) => theme.colors.secondaryColor};
      }

      h4 {
        color: ${({ theme }) => theme.colors.lightGrayColor};
      }

      a {
        color: ${({ theme }) => theme.colors.accentColor};
      }
    `,
  },
};
