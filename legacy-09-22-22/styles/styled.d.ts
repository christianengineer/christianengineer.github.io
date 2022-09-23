import { FlattenInterpolation } from 'styled-components';

declare module 'styled-components' {
  export interface DefaultTheme {
    [key: string]: FlattenInterpolation;

    colors: {
      primaryColor: string;
      secondaryColor: string;
      lightGrayColor: string;
      accentColor: string;
      accentSecondaryColor: string;
    };

    fonts: {
      primaryFont: string;
    };

    gradients: {
      light: string;
      dark: string;
    };

    breakPoints: {
      phoneSmall: string;
      phoneMedium: string;
      phoneLarge: string;
      iPadMedium: string;
      iPadLarge: string;
      smallLaptop: string;
      largeLaptop: string;
      smallDesktop: string;
      mediumDesktop: string;
      largeDesktop: string;
      extraLargeDesktop: string;
      superExtraLargeDesktop: string;
    };
  }
}
