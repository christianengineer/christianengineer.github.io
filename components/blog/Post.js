import styled from 'styled-components';

const Post = styled.div`
  padding: 10rem;

  h1 {
    color: ${({ theme }) => theme.colors.accentColor};
    font-size: 8rem;
  }

  p {
    font-family: Verdana, Geneva, Tahoma, sans-serif;
    font-size: 2rem;
    color: ${({ theme }) => theme.colors.lightGrayColor};
  }
`;

export default Post;
