# tamarindncilantro

1. Importing the library
	from transmission_graph import regions as tg

2. Creating a new transmission graph object: 
	
	g= tg.transmission_graph()

This will create an arbitrary graph with 9 nodes named 'A' to 'I' at arbitrary locations.


3. Adding a new node
	
	#Call the function on the object:-
	
	g.add_node(node_name, x, y, t1, r1, h1)
	
PARAMETERS:-
	node_name - name of the city	(string)
	x- x coordinate of city		(float)
	y- y coordinate of city		(float)
	t1- temperature of the city	(float)
	r1- rainfall of the city	(float)
	h1- humidity of the city	(float)

4. Removing a node

	g.remove_node(node_name)	(string)

5. Modifying the edge weight between two nodes:-

	g.change_weight(r1, r2, wt)

PARAMETERS:-
	r1- region 1 (node 1, ie , source)			(string)
	r2- region 2 (node 2, ie, destination)			(string)
	wt- edge weight you want between these two nodes	(float)

6. Print the graph and susceptibilities of each node:-

	g.print_graph()