import styled from 'styled-components';

type SectionType = {
  readonly sectionBackground: string;
  readonly sectionTheme: string;
  readonly align: string;
};

export const Section = styled.section<SectionType>`
  text-align: ${({ align }) => align};
  position: relative;
  display: flex;

  flex-direction: column;
  padding: 10%;

  ${({ sectionBackground, theme }) =>
    theme.sectionBackgrounds[sectionBackground]}

  background-size: cover;

  div {
    margin-bottom: 40px;
  }

  h2 {
    font-size: 10vw;
    text-transform: uppercase;
  }

  h3 {
    font-size: 5vw;
  }

  h4 {
    font-size: 3vw;
    font-weight: 300;
  }

  a {
    text-decoration: underline;
    font-weight: 400;
  }

  ${({ sectionTheme, theme }) => theme.sectionThemes[sectionTheme]}

  span {
    color: ${({ theme }) => theme.colors.secondaryColor};
    font-weight: 400;
  }
`;
