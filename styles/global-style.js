import { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
  html {
    background-color: ${({ theme }) => theme.colors.primaryColor};
    box-sizing: border-box;
    font-size: 62.5%;
  }

  body {
    font-family: ${({ theme }) => theme.fonts.primaryFont};
  }

  *,
  *:before,
  *:after {
    box-sizing: inherit;
  }

  body,
  h1,
  h2,
  h3,
  h4,
  h5,
  h6,
  ul,
  ol,
  li,
  p,
  pre,
  blockquote,
  figure,
  hr {
    margin: 0;
    padding: 0;
  }
`;

export default GlobalStyle;
