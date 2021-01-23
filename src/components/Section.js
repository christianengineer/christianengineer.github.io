import styled, { css } from 'styled-components';

const backgroundGradients = {
  education: css`
    background: linear-gradient(rgba(58, 55, 41, 0.8), rgba(58, 55, 41, 0.8)),
      url(./images/christian-ipanaque-education.jpg);
  `,
};

const sectionThemes = {
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
  },
};

const linkThemes = {
  light: css`
    text-decoration: underline;
    font-weight: 400;
    color: ${({ theme }) => theme.colors.primaryColor};
  `,
  dark: css`
    text-decoration: underline;
    font-weight: 400;
    color: ${({ theme }) => theme.colors.accentColor};
  `,
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
    ${({ linkTheme }) => linkThemes[linkTheme]}
  }
`;

export default Section;
