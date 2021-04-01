GEOMETRY_LX = 2*Pi;
GEOMETRY_LY = 10;
s1 = .1;
s2 = 3;

// Define domain
Point(1) = {- GEOMETRY_LX/2 , - GEOMETRY_LY/2 , 0 , s1};
Point(2) = {+ GEOMETRY_LX/2 , - GEOMETRY_LY/2 , 0 , s2};
Point(3) = {+ GEOMETRY_LX/2 , + GEOMETRY_LY/2 , 0 , s2};
Point(4) = {- GEOMETRY_LX/2 , + GEOMETRY_LY/2 , 0 , s1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Periodic Line {2} = {4} Translate {GEOMETRY_LX, 0, 0};
// Periodic Line {2} = {4} Translate {0, 0, 0};
Physical Point("pbottom") = {1, 2};
Physical Point("ptop") = {3, 4};
Physical Curve ("periodic") = {2, 4};
Physical Curve ("bottom") = {1};
Physical Curve ("top") = {3};
Physical Surface ("Surface") = {1};

// View options
Geometry.LabelType = 2;
Geometry.Lines = 1;
Geometry.LineNumbers = 2;
Geometry.Surfaces = 1;
Geometry.SurfaceNumbers = 2;

Mesh.RecombineAll=1;
// RecombineMesh;
