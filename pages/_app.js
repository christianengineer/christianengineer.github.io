import '../styles/global-fonts.css';
import Head from 'next/head';
import { ThemeProvider } from 'styled-components';
import theme from '../styles/theme';
import GlobalStyle from '../styles/global-styles';

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <Head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta httpEquiv="X-UA-Compatible" content="ie=edge" />
        <title>Christian Ipanaque - Software Engineer in Seattle, WA</title>
      </Head>
      <Component {...pageProps} />
    </ThemeProvider>
  );
}
