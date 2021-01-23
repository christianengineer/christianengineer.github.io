import styled, { css } from 'styled-components';

const backgroundGradients = {
  education: css`
    background: linear-gradient(rgba(58, 55, 41, 0.8), rgba(58, 55, 41, 0.8)),
      url(./images/christian-ipanaque-education.jpg);
  `,
  achievements: css`
    background: linear-gradient(
        rgba(255, 249, 90, 0.8),
        rgba(255, 249, 90, 0.8)
      ),
      url(./images/christian-ipanaque-achievements.jpg);
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
      font-weight: 400;
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
      font-weight: 400;
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
`;

export default Section;
