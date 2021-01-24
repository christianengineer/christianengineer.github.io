import Head from 'next/head';
import { ThemeProvider } from 'styled-components';
import theme from '../src/styles/theme';
import GlobalStyle from '../src/styles/global-style';

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <Head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta http-equiv="X-UA-Compatible" content="ie=edge" />
        <title>Christian Ipanaque - Software Engineer in Seattle, WA</title>
      </Head>
      <Component {...pageProps} />
    </ThemeProvider>
  );
}
