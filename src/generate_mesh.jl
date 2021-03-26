# Potential
V(q) = (1 - cos(q))/2;
V¯¹(x) = acos(1 - 2*x)
Vmin, Vmax = V(0), V(π)


# First index = q, second index = E
# nodes = [(q, E) for q in nodesq, E in nodesE]
function map2qp(q, E)
  factor = 1
  if E <= 0
    E = -E
    factor = -1
  end
  epsilon = Vmax/10
  # epsilon = 0
  if E >= Vmax + epsilon
    p = sqrt(2*(E - V(q)))
  else
    p = E/(Vmax+epsilon) * sqrt(2*(Vmax + epsilon - V(q)))
    # p = sqrt(max(0, 2*(E - V(q))))
  end
  return (q, factor*p)
end


# function sizeE(e)
#     if e < Vmax
#         return Vmax/40
#     elseif e < 2*Vmax
#         return Vmax/100
#     else
#         return 
# end
# Emax = 10
# fsize(x) = .2*x + Emax*sign(x)*x^10

# nodesE = f.(LinRange(-1, 1, nE + 1))
Emax, nE = 10, 800
# nodesE = (x -> x + x^2*sign(x)).(LinRange(-sqrt(Emax), sqrt(Emax), nE + 1))
nodesE = LinRange(-Emax, Emax, nE + 1)
nq = 200
nodesq = LinRange(-pi, pi, nq + 1)

# nodes = [(q, sqrt(2*E - V(q))) for q in nodesq, E in nodesE]
nodes = [map2qp(q, E) for q in nodesq, E in nodesE];
ind = reshape(1:length(nodes), size(nodes));


# Point elements
# ==============
nelem_points = 4;
elem_points = [ind[1, 1], ind[end, 1],
               ind[end, end], ind[1, end]];

# Elements at the edges
# =====================
nelem_edges = 2*sum(size(nodes) .- 1);
elem_edges1 = zeros(Int, length(nodesq) - 1, 2);
elem_edges2 = zeros(Int, length(nodesE) - 1, 2);
elem_edges3 = zeros(Int, length(nodesq) - 1, 2);
elem_edges4 = zeros(Int, length(nodesE) - 1, 2);

for i in 1:(length(nodesq) - 1)
  elem_edges1[i, :] = [ind[i, 1], ind[i+1, 1]]
  elem_edges3[i, :] = [ind[end+1-i, end], ind[end-i, end]]
end

for i in 1:(length(nodesE) - 1)
  elem_edges2[i, :] = [ind[end, i], ind[end, i+1]]
  elem_edges4[i, :] = [ind[1, end+1-i], ind[1, end-i]]
end

# Elements in the volume
# ======================
nelem_volume = prod(size(nodes) .- 1)
elems = zeros(Int, nelem_volume, 4);
ielem = 1
for i in 1:length(nodesE) - 1
  for j in 1:length(nodesq) -1
    global ielem
    elems[ielem, :] = [ind[j, i], ind[j+1, i],
                       ind[j+1, i+1], ind[j, i+1]];
    ielem += 1
  end
end

# Print file
# ==========

qmin, qmax = nodes[1, 1][1], nodes[end, 1][1]
pmin, pmax = nodes[1, 1][2], nodes[1, end][2]

# Beginning of gmsh file
file = """
\$MeshFormat
4.1 0 8
\$EndMeshFormat
\$PhysicalNames
6
0 1 "pbottom"
0 2 "ptop"
1 3 "periodic"
1 4 "bottom"
1 5 "top"
2 6 "Surface"
\$EndPhysicalNames
\$Entities
4 4 1 0
1 $qmin $pmin 0 1 1
2 $qmax $pmin 0 1 1
3 $qmax $pmax 0 1 2
4 $qmin $pmax 0 1 2
1 $qmin $pmin 0 $qmax $pmin 0 1 4 2 1 -2
2 $qmax $pmin 0 $qmax $pmax 0 1 3 2 2 -3
3 $qmin $pmax 0 $qmax $pmax 0 1 5 2 3 -4
4 $qmin $pmin 0 $qmin $pmax 0 1 3 2 4 -1
1 $qmin $pmin 0 $qmax $pmax 0 1 6 4 1 2 3 4
\$EndEntities
""";

# Add corner nodes
file *= """
\$Nodes
9 $(length(nodes)) 1 $(length(nodes))
0 1 0 1
$(ind[1, 1])
$qmin $pmin 0
0 2 0 1
$(ind[end, 1])
$qmax $pmin 0
0 3 0 1
$(ind[end, end])
$qmax $pmax 0
0 4 0 1
$(ind[1, end])
$qmin $pmax 0""";

function add_edge_nodes(indices, tag)
  global file
  result = "\n1 $tag 0 $(length(indices))"
  for i in indices
    result *= "\n$i"
  end
  for i in indices
    result *= "\n$(nodes[i][1]) $(nodes[i][2]) 0"
  end
  file *= result
end

# Add nodes on edges
edge1 = ind[2:end-1, 1];
edge2 = ind[end, 2:end-1];
edge3 = reverse(ind[2:end-1, end]);
edge4 = reverse(ind[1, 2:end-1]);

add_edge_nodes(edge1, 1);
add_edge_nodes(edge2, 2);
add_edge_nodes(edge3, 3);
add_edge_nodes(edge4, 4);

# Add nodes in volume
function print_inner_nodes()
  ind_volume = ind[2:end-1, 2:end-1]
  result = "\n2 1 0 $(length(ind_volume))"
  for i in ind_volume
    result *= "\n$i"
  end
  for i in ind_volume
    result *= "\n$(nodes[i][1]) $(nodes[i][2]) 0"
  end
  return result
end

file *= print_inner_nodes();
file *= "\n\$EndNodes";

# Add elements
# ============
nelems = nelem_points + nelem_edges + nelem_volume;

file *= """\n\$Elements
9 $nelems 1 $nelems
0 1 15 1
1 $(ind[1, 1])
0 2 15 1
2 $(ind[end, 1])
0 3 15 1
3 $(ind[end, end])
0 4 15 1
4 $(ind[1, end])""";
ielem = 5

function add_edge_elems(elems, tag)
  global file, ielem
  result = "\n1 $tag 1 $(size(elems)[1])"
  for e in eachrow(elems)
    result *= "\n$ielem $(e[1]) $(e[2])"
    ielem += 1
  end
  file *= result
end

add_edge_elems(elem_edges1, 1);
add_edge_elems(elem_edges2, 2);
add_edge_elems(elem_edges3, 3);
add_edge_elems(elem_edges4, 4);

# Tag = 1
function add_volume_elems(elems)
  global file, ielem
  file *= "\n2 1 3 $(size(elems)[1])"
  for e in eachrow(elems)
    file *= "\n$ielem $(e[1]) $(e[2]) $(e[3]) $(e[4])"
    ielem += 1
  end
end

add_volume_elems(elems);
file *= "\n\$EndElements";

# Periodic
# ========

file *= "\n\$Periodic
3
0 2 1
16 1 0 0 $(2π) 0 1 0 0 0 0 1 0 0 0 0 1
1
$(ind[end, 1]) $(ind[1, 1])
0 3 4
16 1 0 0 6.283 0 1 0 0 0 0 1 0 0 0 0 1
1
$(ind[end, end]) $(ind[1, end])
1 2 4
16 1 0 0 6.283 0 1 0 0 0 0 1 0 0 0 0 1
$(size(nodes)[2])";

for i in 1:size(nodes)[2]
   global file
   file *= "\n$(ind[end, i]) $(ind[1, i])";
end
file *= "\n\$EndPeriodic";

# Write to file
# =============
open("mymesh.msh", "w") do io
   write(io, file)
end;
