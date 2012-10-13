-------------------------------------------------------------------------------
CIS565: Project 2: CUDA Pathtracer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Friday, 10/12/2012
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Path Tracer
-------------------------------------------------------------------------------
For this project I've extended the ray tracer into a path tracer. All of the base requirements have been met and I have added in camera controls and a fancy new primitive: the portal!

Camera controls demoed in the video.

Portals, like the ones in the scene are comprised of a planar colliding surface defined by a Julia set the parameters of which are defined in the scene file (circles are a special case of the Julia set with C set to 0). Each portal is linked to another portal, allowing for a ray to enter portal A and exit portal B. I have added the functionality that input and output portals need not be the same size, this allows for scaling effects that make objects appear larger or smaller or in some way warped. Portals may map to any other portal. This means that you can have one portal be the output of many different inputs, i.e. portals A, B, and C all lead to portal D. I use this characteristic to create the back lighting on the circular and fractal portals in the scene by adding a duplicate of the portals behind the main viewing ones that all map to a portal placed outside of the bounding box facing a light source.

As can be seen by comparing this video to the last, I have upgraded my hardware to an Nvidia GTX 680 and have seen substantial speed boosts.

-------------------------------------------------------------------------------
BLOG:
-------------------------------------------------------------------------------
http://liamboone.blogspot.com/2012/10/project-2-pathtracer.html