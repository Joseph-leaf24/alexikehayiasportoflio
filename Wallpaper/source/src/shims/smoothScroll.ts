/**
 * What the scene wants from the site's smooth scrolling: only whether to hold still for
 * "reduced motion". A wallpaper never does (Windows turns reduced motion on with its "show
 * animations" setting, which would freeze the desktop on one frame).
 */
export const prefersReducedMotion = () => false;
