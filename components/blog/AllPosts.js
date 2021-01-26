import styled from 'styled-components';

const AllPosts = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;

  h2 {
    text-transform: uppercase;
    color: ${({ theme }) => theme.colors.accentColor};
    font-size: 8rem;
  }

  p {
    font-family: Verdana, Geneva, Tahoma, sans-serif;
    font-size: 2rem;
    color: ${({ theme }) => theme.colors.lightGrayColor};
  }
`;

const Post = styled.div`
  padding: 12rem;
`;

AllPosts.Post = Post;

export default AllPosts;
