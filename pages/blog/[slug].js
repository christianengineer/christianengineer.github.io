import { getAllPosts, getPostBySlug, markdownToHtml } from 'lib/blog';
import Head from 'next/head';
import { Post as StyledPost } from '@blog-components';

export default function Post({ post: { title, content } }) {
  return (
    <article>
      <Head>
        <title>{title} | Christian Ipanaque</title>
      </Head>
      <StyledPost>
        <h1>{title}</h1>
        <p
          dangerouslySetInnerHTML={{
            __html: content
          }}
        />
      </StyledPost>
    </article>
  );
}

export async function getStaticProps({ params }) {
  const { title, content } = getPostBySlug(params.slug, ['title', 'content']);
  const htmlContent = await markdownToHtml(content || '');

  return {
    props: {
      post: {
        title,
        content: htmlContent
      }
    }
  };
}

export async function getStaticPaths() {
  const posts = getAllPosts(['slug']);

  return {
    paths: posts.map(({ slug }) => {
      return {
        params: {
          slug
        }
      };
    }),
    fallback: false
  };
}
