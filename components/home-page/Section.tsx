import styled from 'styled-components';

type SectionType = {
  backgroundGradient: string;
  sectionTheme: string;
  align: string;
};

export const Section = styled.section<SectionType>`
  text-align: ${({ align }) => align};
  position: relative;
  display: flex;

  flex-direction: column;
  padding: 10%;
  ${({ backgroundGradient, theme }) =>
    theme.backgroundGradients[backgroundGradient]}
  background-size: cover;

  div {
    margin-bottom: 40px;
  }

  h2 {
    font-size: 10vw;
    text-transform: uppercase;
    ${({ sectionTheme, theme }) => theme.sectionThemes[sectionTheme]['h2']}
  }

  h3 {
    font-size: 5vw;
    ${({ sectionTheme, theme }) => theme.sectionThemes[sectionTheme]['h3']}
  }

  h4 {
    font-size: 3vw;
    font-weight: 300;
    ${({ sectionTheme, theme }) => theme.sectionThemes[sectionTheme]['h4']}
  }

  a {
    ${({ sectionTheme, theme }) => theme.sectionThemes[sectionTheme]['a']}
  }

  span {
    color: ${({ theme }) => theme.colors.secondaryColor};
    font-weight: 400;
  }
`;
