import styled, { css } from 'styled-components';

interface NavProps {
  readonly top: boolean;
  readonly bottom: boolean;
}

export const Nav = styled.nav<NavProps>`
  display: none;
  position: absolute;
  justify-content: center;
  opacity: 0;
  font-size: 1.6rem;
  animation: 1.2s ease-out 2s forwards fadeIn;
  z-index: 1;

  ${({ top }: NavProps) =>
    top &&
    css`
      margin-top: 40px;
      right: 30%;
    `}

  ${({ bottom }: NavProps) =>
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

export const NavLink = styled.a`
  color: inherit;
  text-decoration: none;
  padding: 15px;
  border-bottom: 1px solid transparent;
  border-left: 1px solid transparent;
  transition: 0.5s ease-in-out;

  &:hover {
    cursor: pointer;
    border-bottom: 1px solid ${({ theme }) => theme.colors.primaryColor};
    border-left: 1px solid ${({ theme }) => theme.colors.primaryColor};
  }
`;
