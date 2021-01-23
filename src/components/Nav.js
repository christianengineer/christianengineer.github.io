import styled, { css } from 'styled-components';

const Nav = styled.nav`
  display: none;
  position: absolute;
  justify-content: center;
  opacity: 0;
  font-size: 1.6rem;
  animation: 1.2s ease-out 2s forwards fadeIn;
  z-index: 1;
  ${({ top }) =>
    top &&
    css`
      margin-top: 40px;
      right: 30%;
    `}

  ${({ bottom }) =>
    bottom &&
    css`
      margin-bottom: 40px;
      margin-left: 45px;
      bottom: 0;
    `}

  div {
    padding-right: 10px;

    @media (min-width: ${({ theme }) => theme.breakPoints.phoneLarge}) {
      padding-right: 50px;
    }
  }

  @media (min-width: ${({ theme }) => theme.breakPoints.phoneMedium}) {
    display: flex;
  }
`;

export default Nav;
