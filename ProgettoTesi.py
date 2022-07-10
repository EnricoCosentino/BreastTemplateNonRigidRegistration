import open3d as o3d
import numpy as np
import openmesh as om
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sksparse.cholmod import cholesky_AAt
from scipy import sparse
import os
from Graph import Graph_struct
import pymeshlab
# import nicp_meshes

def spsolve_chol(sparse_X, dense_b):
    factor = cholesky_AAt(sparse_X.T)
    return factor(sparse_X.T.dot(dense_b)).toarray()

def calculate_rotation_matrix(vectorA, vectorB):
    vectorA = (vectorA/np.linalg.norm(vectorA)).reshape(3)
    vectorB = (vectorB/np.linalg.norm(vectorB)).reshape(3)
    rotAxis = np.cross(vectorA, vectorB)
    angleSin = np.linalg.norm(rotAxis)
    angleCos = np.dot(vectorA, vectorB)
    mat = np.matrix([[    0      , -rotAxis[2],  rotAxis[1]],\
                     [ rotAxis[2],      0     , -rotAxis[0]],\
                     [-rotAxis[1],  rotAxis[0],      0     ],\
                    ])
    rotMatrix = np.identity(3) + mat + mat.dot(mat)*((1-angleCos)/(angleSin**2))
    return rotMatrix

def rejoin_mesh(vertices, faces):
    mesh = om.TriMesh()
    vertices_handles = []
    for vertex in vertices:
        vertices_handles.append(mesh.add_vertex(vertex))
    for face in faces:
        mesh.add_face(vertices_handles[face[0]], vertices_handles[face[1]], vertices_handles[face[2]])
    return mesh

def write_mesh(vertices, faces, path):
    mesh = rejoin_mesh(vertices, faces)
    om.write_mesh(path, mesh)
    
def center_mesh(vertices):
    center = np.mean(vertices, axis = 0)
    return vertices - center

def fit_mesh(vertices):
    mostProtrudingVertexIndex = 0
    mostProtrudingVertexAxis = 0
    for i in range(len(vertices)):
        for axis, position in enumerate(vertices[i]):
            if np.abs(vertices[mostProtrudingVertexIndex][mostProtrudingVertexAxis]) < np.abs(vertices[i][axis]):
                mostProtrudingVertexIndex = i
                mostProtrudingVertexAxis = axis  
    scaleFactor = 1 / np.abs(vertices[mostProtrudingVertexIndex][mostProtrudingVertexAxis])
    return vertices * scaleFactor

#links si riferisce a spigoli o facce
def recalculate_links(oldLinks, newIndices):
    newFaces = [x for x in oldLinks if x[0] in newIndices]
    newFaces = np.array(newFaces)
    return newFaces

def map_vertices(oldVertices, newVertices):
    pdVerticesOld = pd.DataFrame(oldVertices)
    pdVerticesNew = pd.DataFrame(newVertices)
    pdVerticesOld.reset_index(inplace = True)
    pdVerticesNew.reset_index(inplace = True)
    verticesMap = pd.merge(pdVerticesOld, pdVerticesNew, on = [0,1,2], how="inner")
    verticesMap.rename(columns = {"index_x":"OldIndex", "index_y":"NewIndex"}, inplace = True)
    verticesMap.drop(columns = [0,1,2], inplace = True)
    return verticesMap.values

def rename(nameMap, array):
    nameMap = np.array([x for x in nameMap if x[0] != x[1]])
    for element in nameMap:
        array[array == element[0]] = element[1]
    return array

def map_surfaces(oldVertices, oldEdges, oldFaces, newIndices):
    newVertices = oldVertices[newIndices]
    vertexMap = map_vertices(oldVertices, newVertices)
    newEdges = recalculate_links(oldEdges, newIndices)
    newEdges = rename(vertexMap, newEdges)
    newFaces = recalculate_links(oldFaces, newIndices)
    newFaces = rename(vertexMap, newFaces)
    return newVertices, newEdges, newFaces

def clean_mesh(vertices, edges, faces):
    graph = Graph_struct(len(vertices))
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    connComps = graph.connected_components()
    connComps = [np.array(x) for x in connComps]
    largestConnComp = np.sort(max(connComps, key = len))
    return map_surfaces(vertices, edges, faces, largestConnComp)

# dataset_dir = "Mesh/dataset_raw/Database_RAW"
# for directory in os.listdir(dataset_dir):
source_filename = "Template1030"
source_path = "Mesh/" + source_filename + ".obj"
# target_filename = "Model"
# target_path = dataset_dir + "/" + directory + "/" + target_filename + ".obj"
target_filename = "CutTriModelRot"
target_path = "Mesh/dataset_raw/Database4_RAW/02_Model/" + target_filename + ".obj"


##TARGET IMPORT, SCALING AND CLEANING##
target = o3d.io.read_triangle_mesh(target_path)
target.remove_duplicated_vertices()
target.remove_degenerate_triangles()
targetVertices = np.asarray(target.vertices)

targetEdges = []
targetFaces = np.asarray(target.triangles)
for face in targetFaces:
    sortedFace = np.sort(face)
    targetEdges.append(tuple(sortedFace[1:]))
    targetEdges.append(tuple(sortedFace[:2]))
targetEdges = set(targetEdges)

cleanedTarget = clean_mesh(targetVertices, targetEdges, targetFaces)
targetVertices = cleanedTarget[0]
targetFaces = cleanedTarget[2]

targetVertices = center_mesh(targetVertices)
# Scaling ignorato
# targetVertices = fit_mesh(targetVertices)

target.vertices = o3d.utility.Vector3dVector(targetVertices)
target.triangles = o3d.utility.Vector3iVector(targetFaces)
targetPath = "Mesh/clean_target.obj"
o3d.io.write_triangle_mesh(targetPath, target)

#ms contiene clean_scaled_target
ms = pymeshlab.MeshSet()
ms.load_new_mesh(targetPath)

#ms contiene clean_scaled_target_closed_holes
try: 
    ms.apply_filter("meshing_close_holes")
except: 
    print("Could not close holes in target")
    # ms.generate_surface_reconstruction_screened_poisson()
    # targetPath = "Mesh/poisson_target.obj"
    
targetPath = "Mesh/clean_target_closed_holes.obj"
ms.save_current_mesh(targetPath)
target = o3d.io.read_triangle_mesh(target_path)
targetVertices = np.asarray(target.vertices)
targetFaces = np.asarray(target.triangles)

# cleanScaledFilledTarget = mc.Mesh(ms.current_mesh().vertex_matrix(),\
#                                   ms.current_mesh().face_matrix())
dist_function = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(ms.current_mesh().vertex_matrix())

##SOURCE IMPORT, CLEANING AND ALIGNMENT##
source = o3d.io.read_triangle_mesh(source_path)
source.remove_duplicated_vertices()
source.remove_degenerate_triangles()

sourceEdges = []
sourceFaces = np.asarray(source.triangles)
sourceVertices = np.asarray(source.vertices)
for face in sourceFaces:
    sortedFace = np.sort(face)
    sourceEdges.append(tuple(sortedFace[1:]))
    sourceEdges.append(tuple(sortedFace[:2]))
sourceEdges = set(sourceEdges)

# cleanedSource = clean_mesh(sourceVertices, sourceEdges, sourceFaces)
# sourceVertices = cleanedSource[0]
# sourceEdges = cleanedSource[1]
# sourceFaces = cleanedSource[2]

# sourceEdges = set([tuple(i) for i in sourceEdges])
e = len(sourceEdges)
sourceVertices = center_mesh(sourceVertices)
# Anche qua ignoriamo lo scaling
# sourceVertices = fit_mesh(sourceVertices)
n = len(sourceVertices)

source.vertices = o3d.utility.Vector3dVector(sourceVertices)
# source.triangles = o3d.utility.Vector3iVector(sourceFaces)
o3d.io.write_triangle_mesh("Mesh/centered_source.obj", source)

sourceNormals = np.asarray(source.vertex_normals)
targetNormals = np.asarray(target.vertex_normals)
meanSourceNormal = np.mean(sourceNormals, axis=0).T
meanTargetNormal = np.mean(targetNormals, axis=0).T
if np.count_nonzero(np.isnan(targetNormals)) == 0:
    rotMatrix = calculate_rotation_matrix(meanSourceNormal, meanTargetNormal)
    sourceVertices = np.apply_along_axis(lambda x: rotMatrix @ x, 1, sourceVertices)
    source.vertices = o3d.utility.Vector3dVector(sourceVertices)
    o3d.io.write_triangle_mesh("Mesh/centered_aligned_source.obj", source)

##FINE TRATTAMENTO SOURCE##

#ms contiene poisson_target
# ms.apply_filter("generate_surface_reconstruction_screened_poisson", preclean=True)
# targetPath = "Mesh/poisson_target.obj"
# ms.save_current_mesh(targetPath)
# poissonDists, poissonInds = dist_function.kneighbors(ms.current_mesh().vertex_matrix())
# acceptedVertices = poissonDists <= 0.04
# newTargetIndices = np.where(acceptedVertices == True)[0]

#Operazione piuttosto lenta a causa del numero di vertici.
#Funziona sui vertici ma non sulle facce
#Far precedere una voxelization?
# cleanedPoissonTarget = map_surfaces(ms.current_mesh().vertex_matrix(),\
#                                     ms.current_mesh().edge_matrix(),\
#                                     ms.current_mesh().face_matrix(),\
#                                     newTargetIndices)
# targetVertices = cleanedPoissonTarget[0]
# targetEdges = cleanedPoissonTarget[1]
# targetFaces = cleanedPoissonTarget[2]

# target.vertices = o3d.utility.Vector3dVector(targetVertices)
# target.triangles = o3d.utility.Vector3iVector(targetFaces)
# targetPath = "Mesh/clean_poisson_target.obj"
# o3d.io.write_triangle_mesh(targetPath, target)
# poissonTarget = mc.Mesh(ms.current_mesh().vertex_matrix(),\
#                         ms.current_mesh().face_matrix())
# test = mc.generation.thicken(cleanScaledFilledTarget, 0.01)
    
M = sparse.lil_matrix((e, n), dtype=np.float32)

for i, t in enumerate(sourceEdges):
    M[i, t[0]] = -1
    M[i, t[1]] = 1

gamma = 1
G = np.diag([1, 1, 1, gamma]).astype(np.float32)

Es = sparse.kron(M, G)
print("Es shape:", Es.shape)

nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(targetVertices)
nbrs50 = NearestNeighbors(n_neighbors=30, algorithm='kd_tree').fit(targetVertices)
# landmark_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sourceVertices[:,:3])

alphas = [50,35,20,15,7,3,2,1] #stiffness terms
# alphas = [i/2 for i in alphas]
betas = [15,14, 3, 2, 0.5, 0,0,0] #landmark weights
# landmarks = mpp.load("Mesh/PickedPoints1.pp")
# landmark_points = []
# for landmark in landmarks:
#     landmark_points.append(landmark['point'])
# landmark_points = np.asarray(landmark_points)
# landmark_points = center_mesh(landmark_points)
# landmark_points = fit_mesh(landmark_points)
# nLandmarks = landmark_points.shape[0]
# landmark_dists, landmark_indices = landmark_nbrs.kneighbors(landmark_points)

for cnt,(alpha_stiffness,beta) in enumerate(zip(alphas,betas)):

    # print("using alpha stiffness and beta(lm wt): ",alpha_stiffness,beta)

    iter_max = 5
    if(alpha_stiffness<25):
        iter_max = 3
        
    for iter_cnt in range(iter_max):

        #####DATA WEIGHTS########
        weights = np.ones([n,1])
        if(alpha_stiffness>25):
            weights*=0.5
        
        if(alpha_stiffness>15): #no normals used
            distances, indices = nbrs.kneighbors(sourceVertices[:,:3])
            print("median mesh distance:",np.median(distances))
            indices = indices.squeeze()
            
            matches = targetVertices[indices]
        else:
            indices = None
                
            distances = np.zeros(sourceVertices.shape[0])
            matches = np.zeros_like(sourceVertices)
            source.compute_vertex_normals()
            mesh_norms = np.asarray(source.triangle_normals)
            dist50, inds50 = nbrs50.kneighbors(sourceVertices)
            eps = 0.1  #remember, rescaling system to 1
            
            for i in range(sourceVertices.shape[0]):
                pt_normal = mesh_norms[i]
                corr50 = targetVertices[inds50[i]]
                ab = sourceVertices[i] - corr50
                c = np.cross(ab,pt_normal)
                line_dists = np.linalg.norm(c,axis =1)
                # print(np.min(line_dists),np.mean(line_dists),np.max(line_dists))
                fltr = line_dists<eps
                # pdb.set_trace()
                if(np.sum(fltr)>0):
                    matches[i,:]  = np.mean(corr50[fltr],axis=0)
                distances[i] = np.linalg.norm(matches[i] - sourceVertices[i])
                        
        ########## IMPORTANTE ##########
        #Sistemare la parte dei mismatch
        # d_thresh = 0.05 #(np.linalg.norm(sourceVertices[2087, :3] - sourceVertices[14471, :3])) / 4 #Significato indici?
        sourceVertices = np.hstack((sourceVertices, np.ones([n, 1])))
                        
        # mismatches = np.where(distances>d_thresh)[0]
        # weights[mismatches] = 0
        # weights[landmark_indices] = 0
        
        mesh_matches = np.where(weights > 0)[0]
        
        B = sparse.lil_matrix((4 * e + n, 3), dtype=np.float32)
        # DL = sparse.lil_matrix((nLandmarks, 4 * n), dtype=np.float32)

        V = sparse.lil_matrix((n, 4 * n), dtype=np.float32)
        
        B[4 * e: (4 * e + n), :] = weights * matches
        for i in range(n):
            # D[i,4*i:4*i+4] = weights[i]*mesh[i]
            V[i, 4 * i:4 * i + 4] = sourceVertices[i]
            
        D = V.multiply(weights)

        # lm_wtmat = beta * np.ones([DL.shape[0], 1])

        # for i, lm in enumerate(landmark_indices):
        #     DL[i, (4 * lm)[0]:(4 * lm + 4)[0]] = lm_wtmat[i] * sourceVertices[lm]  ##BETA moved here !!
            
        D = D.tocsr()
        # DL = DL.tocsr()
        A = sparse.csr_matrix(sparse.vstack([alpha_stiffness * Es, D,])) #DL]))
        
        # B = sparse.vstack((B, lm_wtmat * landmark_points))  ##assuming typo in paper, beta should be weighing both ?
        #print("B size after lms", B.shape)
        #print("solving...")
        X = spsolve_chol(A, B)
        
        #print("warping...")
        new_verts = V.dot(X)  ##X from spsolve_chol is Dense already
        
        # if (iter_cnt == iter_max - 1) and alpha_stiffness == 1:
        if (iter_cnt == iter_max - 1):
            print("saving...")
            new_mesh_path = 'Mesh/Results/deformed_alpha_' + str(alpha_stiffness) + '_' + target_filename + '.obj'
            # new_mesh_path = dataset_dir + '/Results/' + directory + '.obj'
            # if (alpha_stiffness == 15):
            #     new_mesh_path = 'Mesh/final_mesh.obj'
            # vs = np.asarray(new_verts) #* scale_factor + centre
            # if (alpha_stiffness <= 15):
            #     match_mesh = vs[mesh_matches]
            #     targ_matches = (matches[np.where(weights > 0)[0]]) #* scale_factor + centre
            #     write_ply(seq + '/matched_alpha_' + str(alpha_stiffness) + '_mesh.ply', match_mesh, normals=None)
            #     write_ply(seq + '/matched_alpha_' + str(alpha_stiffness) + '_target.ply', targ_matches, normals=None)
            o3d.io.write_triangle_mesh(new_mesh_path, source, write_vertex_normals = False,)
        sourceVertices = new_verts
        source.vertices = o3d.utility.Vector3dVector(new_verts)
    
    # new_mesh_path = 'Mesh/reduced_mesh.obj'
    # o3d.io.write_triangle_mesh(new_mesh_path, reducedMesh, write_vertex_normals = False,)
    # ms = pymeshlab.MeshSet()
    # ms.load_new_mesh(new_mesh_path)
    # ms.apply_filter("generate_surface_reconstruction_screened_poisson", preclean=True)
    # new_mesh_path = "Mesh/poisson_mesh.obj"
    # ms.save_current_mesh(new_mesh_path)
    # poisson_mesh = o3d.io.read_triangle_mesh(new_mesh_path)

