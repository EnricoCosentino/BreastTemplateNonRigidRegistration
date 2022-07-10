class Graph_struct:
   def __init__(self, V):
      self.V = V
      self.adj = [[] for i in range(V)]

   def DFS_Utililty(self, temp, v, visited):
      temp.append(v)
      conn_comp = []
      while len(temp):
          s = temp[-1]
          temp.pop()
          if not visited[s]:
              visited[s] = True
              conn_comp.append(s)
          for node in self.adj[s]:
              if not visited[node]:
                  temp.append(node)
      return conn_comp
      # for i in self.adj[v]:
      #    if visited[i] == False:
      #       temp = self.DFS_Utililty(temp, i, visited)
      # return temp

   def add_edge(self, v, w):
      self.adj[v].append(w)
      self.adj[w].append(v)

   def connected_components(self):
      visited = []
      conn_compnent = []
      for i in range(self.V):
         visited.append(False)
      for v in range(self.V):
         if visited[v] == False:
            temp = []
            conn_compnent.append(self.DFS_Utililty(temp, v, visited))
      return conn_compnent

# my_instance = Graph_struct(5)
# my_instance.add_edge(1, 0)
# my_instance.add_edge(2, 3)
# my_instance.add_edge(3, 0)
# print("1-->0")
# print("2-->3")
# print("3-->0")
# conn_comp = my_instance.connected_components()
# print("The connected components are :")
# print(conn_comp)