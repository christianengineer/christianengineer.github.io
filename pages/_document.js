import Document, { Html, Head, Main, NextScript } from 'next/document';

class MyDocument extends Document {
  render() {
    return (
      <Html lang="en">
        <Head>
          <link
            rel="preload"
            as="font"
            type="font/woff2"
            href="/fonts/oswald-v24-latin-regular.woff2"
            crossOrigin="anonymous"
          />

          <link
            rel="preload"
            as="font"
            type="font/woff2"
            href="/fonts/oswald-v24-latin-300.woff2"
            crossOrigin="anonymous"
          />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
