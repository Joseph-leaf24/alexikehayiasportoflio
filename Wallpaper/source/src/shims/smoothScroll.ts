/**
 * Replaces Site 2's smooth-scroll module. The wallpaper never honours "reduced motion": Windows
 * turns it on with its animation setting, which would freeze the desktop on one frame.
 */
export const prefersReducedMotion = () => false;
