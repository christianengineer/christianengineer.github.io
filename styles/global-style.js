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

  @font-face {
    font-family: 'Oswald';
    font-style: normal;
    font-weight: 400;
    font-display: optional;
    src: url('/fonts/oswald-v24-latin-regular.woff2') format('woff2');
  }

  @font-face {
    font-family: 'Oswald';
    font-style: normal;
    font-weight: 300;
    font-display: optional;
    src: url('/fonts/oswald-v24-latin-300.woff2') format('woff2');
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  @keyframes slideInFromLeft {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(0);
    }
  }

  @keyframes slideInFromRight {
    0% {
      transform: translateX(300%);
    }
    100% {
      transform: translateX(0);
    }
  }

  @keyframes slideInFromBottom {
    0% {
      transform: rotate(-90deg) translateX(-100%);
    }
    100% {
      transform: rotate(-90deg) translateX(0);
    }
  }

  @keyframes slideInFromTop {
    0% {
      transform: rotate(-90deg) translateX(200%);
    }
    100% {
      transform: rotate(-90deg) translateX(0);
    }
  }
`;

export default GlobalStyle;
