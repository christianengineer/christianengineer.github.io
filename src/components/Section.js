import styled, { css } from 'styled-components';

const backgroundGradients = {
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
};

const sectionThemes = {
  light: {
    h2: css`
      color: ${({ theme }) => theme.colors.primaryColor};
    `,
    h3: css`
      color: ${({ theme }) => theme.colors.accentSecondaryColor};
    `,
    h4: css`
      color: ${({ theme }) => theme.colors.accentSecondaryColor};
    `,
    a: css`
      text-decoration: underline;
      color: ${({ theme }) => theme.colors.primaryColor};
    `,
  },
  dark: {
    h2: css`
      color: ${({ theme }) => theme.colors.accentColor};
    `,
    h3: css`
      color: ${({ theme }) => theme.colors.secondaryColor};
    `,
    h4: css`
      color: ${({ theme }) => theme.colors.lightGrayColor};
    `,
    a: css`
      text-decoration: underline;
      color: ${({ theme }) => theme.colors.accentColor};
    `,
  },
};

const Section = styled.section`
  text-align: ${({ align }) => align};
  position: relative;
  display: flex;

  flex-direction: column;
  padding: 10%;
  ${({ backgroundGradient }) => backgroundGradients[backgroundGradient]}
  background-size: cover;

  div {
    margin-bottom: 40px;
  }

  h2 {
    font-size: 10vw;
    text-transform: uppercase;
    ${({ sectionTheme }) => sectionThemes[sectionTheme]['h2']}
  }

  h3 {
    font-size: 5vw;
    ${({ sectionTheme }) => sectionThemes[sectionTheme]['h3']}
  }

  h4 {
    font-size: 3vw;
    font-weight: 300;
    ${({ sectionTheme }) => sectionThemes[sectionTheme]['h4']}
  }

  a {
    ${({ sectionTheme }) => sectionThemes[sectionTheme]['a']}
  }

  span {
    color: ${({ theme }) => theme.colors.secondaryColor};
  }
`;

export default Section;
