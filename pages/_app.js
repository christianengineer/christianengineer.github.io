import { ThemeProvider } from 'styled-components';
import theme from '../src/styles/theme';
import GlobalStyle from '../src/styles/global-style';

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <Component {...pageProps} />
    </ThemeProvider>
  );
}
