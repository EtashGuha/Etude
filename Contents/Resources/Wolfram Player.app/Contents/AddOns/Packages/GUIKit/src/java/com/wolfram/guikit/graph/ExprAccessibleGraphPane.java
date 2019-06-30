/*
 * @(#)ExprAccessibleJGraph.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.graph;

import java.awt.Image;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import javax.swing.SwingUtilities;

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.guikit.GUIKitUtils;
import com.wolfram.guikit.type.ExprAccessible;

import com.wolfram.jlink.Expr;
import com.wolfram.jlink.ExprFormatException;
import com.wolfram.jlink.LoopbackLink;
import com.wolfram.jlink.MathLinkFactory;

import diva.canvas.CanvasUtilities;
import diva.canvas.Figure;
import diva.graph.GraphController;
import diva.graph.GraphPane;
import diva.graph.NodeController;
import diva.graph.basic.BasicGraphModel;
import diva.graph.basic.BasicLayoutTarget;
import diva.graph.layout.GridAnnealingLayout;
import diva.graph.layout.LevelLayout;
import diva.util.PropertyContainer;

/**
 * ExprAccessibleGraphPane extends GraphPane
 */
public class ExprAccessibleGraphPane extends GraphPane implements ExprAccessible {

  private MathematicaBSFEngine engine = null;
	
	protected static final Expr SYM_GRAPH = new Expr(Expr.SYMBOL, "Graph");
	protected static final Expr SYM_ARRAYRULES = new Expr(Expr.SYMBOL, "ArrayRules");
	protected static final Expr SYM_RULE = new Expr(Expr.SYMBOL, "Rule");
	protected static final Expr SYM_OFF = new Expr(Expr.SYMBOL, "Off");
	
	public ExprAccessibleGraphPane(BasicGraphModel model) {
		super(model);
	  }
  
	/** Create a new graph pane with the given controller and model.
	 */
	public ExprAccessibleGraphPane(GraphController controller, BasicGraphModel model) {
		super(controller, model);
	  }
	  
  public void setEngine(MathematicaBSFEngine e) {
    engine = e;
    }
    
	public Expr getExpr() {
		PropertyContainer root = (PropertyContainer)getGraphModel().getRoot();
		if (root.getProperty("defaultValue") != null)
			return getArrayRulesExpr((BasicGraphModel)getGraphModel(), getGraphModel().getRoot());
	  else 
	    return getGraphExpr((BasicGraphModel)getGraphModel(), getGraphModel().getRoot());
		}
		
	public Expr getGraphExpr(BasicGraphModel model, Object node) {
		// Walk graph, nodes and edges and create Expr from loopback link
		Expr result = null;
		PropertyContainer root = (PropertyContainer)model.getRoot();
    Boolean rootDirected = Boolean.FALSE;
    if (root.getProperty("directed") != null) {
    	if (root.getProperty("directed").equals(Boolean.TRUE)) rootDirected = Boolean.TRUE;
      }
		try {
			LoopbackLink ml = MathLinkFactory.createLoopbackLink();
			ArrayList nodeList = new ArrayList();
			ml.putFunction("Graph", 3);
				 // put edgeList
				 //		Edges
				 int edgeCount = 0;
				 for (Iterator i = model.nodes(node); i.hasNext(); ) {
				 	  Object n = i.next();
				 	  nodeList.add(n);
						for (Iterator j = model.inEdges(n); j.hasNext(); ) {j.next();
								++edgeCount; } }
				 ml.putFunction("List", edgeCount);
				 for (Iterator i = model.nodes(node); i.hasNext(); ) {
						 Object n = (Object) i.next();
						 for (Iterator j = model.inEdges(n); j.hasNext(); ) {
						 	   // TODO check for edge options
							   ml.putFunction("List", 1);
								 Object e = (Object) j.next();
							   ml.putFunction("List", 2);
							   ml.put(nodeList.indexOf(model.getTail(e))+1);
							   ml.put(nodeList.indexOf(n)+1);
						 }
				   }
				// put NodeList with coordinates
		    int count = nodeList.size();
			  ml.putFunction("List", count);
		    for (int i =0; i <count; ++i) {
          // TODO check for node options
					ml.putFunction("List", 1);
					ml.putFunction("List", 2);
					Figure f = getGraphController().getFigure( nodeList.get(i));
					Point2D loc = CanvasUtilities.getCenterPoint(f);
					ml.put(loc.getX());
					// TODO just a quick fix for a flipped coordinate system
					ml.put(-loc.getY());
		      }
				// put root directed option
			  ml.putFunction("Rule", 2);
			  ml.putSymbol("EdgeDirection");
			  ml.put( rootDirected.equals(Boolean.TRUE));
			ml.flush();
			result = ml.getExpr();
			ml.close();
			}
		catch (Exception ex){}
		
		return result;
		}
		
	public Expr getArrayRulesExpr(BasicGraphModel model, Object node) {
     //	Walk graph, nodes and edges and create Expr from loopback link
		 Expr result = null;
		 PropertyContainer root = (PropertyContainer)model.getRoot();
     Expr defaultValue = (Expr)root.getProperty("defaultValue");
		 try {
			 LoopbackLink ml = MathLinkFactory.createLoopbackLink();
			 ArrayList nodeList = new ArrayList();
  
				// put edgeList
				//		Edges
				int edgeCount = 0;
				for (Iterator i = model.nodes(node); i.hasNext(); ) {
					 Object n = i.next();
					 nodeList.add(n);
					 for (Iterator j = model.inEdges(n); j.hasNext(); ) {j.next();
							 ++edgeCount; } }
				ml.putFunction("List", edgeCount + 1);
				for (Iterator i = model.nodes(node); i.hasNext(); ) {
						Object n = (Object) i.next();
						for (Iterator j = model.inEdges(n); j.hasNext(); ) {
								Object e = (Object) j.next();
								ml.putFunction("Rule", 2);
							  ml.putFunction("List", 2);
								ml.put(nodeList.indexOf(model.getTail(e))+1);
								ml.put(nodeList.indexOf(n)+1);
								Object expr = ((PropertyContainer)e).getProperty("expr");
								if (expr != null)
								  ml.put((Expr)expr);
								else 
								  ml.put(defaultValue);
						}
					}

				 // put defaultValue rule
				 ml.putFunction("Rule", 2);
			     ml.putFunction("List", 2);
			       ml.putFunction("Blank", 0);
			       ml.putFunction("Blank", 0);
			     ml.put(defaultValue);

			 ml.flush();
			 result = ml.getExpr();
			 ml.close();
			 }
		 catch (Exception ex){}
		
		 return result;
		}
		
  // TODO consider a public addExpr that takes graph, nodes, edges etc
  
	public void setExpr( Expr e) {
		if (e == null) return;

    // Before removing existing nodes and edges one must clear selection
		getGraphController().getSelectionModel().clearSelection();
		
		// This removes existing nodes and edges removing nodes in such a way
		// that edges involved are also removed properly
		int count = getGraphModel().getNodeCount(getGraphModel().getRoot());
		Object nodes[] = new Object[count];
		Iterator it = getGraphModel().nodes(getGraphModel().getRoot());
		int i = 0;
    while (it.hasNext()) {
			nodes[i++] = it.next();
      }
		for(int j = 0; j < count; ++j) {
			NodeController nodeControl = getGraphController().getNodeController(nodes[j]);
			nodeControl.removeNode(nodes[j]);
		  }
		
		// Now we can add new content based on expr
		Expr head = e.head();
		if (head != null){
			if (head.equals(SYM_GRAPH)) {
				buildGraph( (BasicGraphModel)getGraphModel(), getGraphModel().getRoot(), e);
			  }
			else if (head.equals(Expr.SYM_LIST)) {
				buildArrayRules( (BasicGraphModel)getGraphModel(), getGraphModel().getRoot(), e);
			  }
		  }

		}

  protected Image createTypesetImage(Expr e) {
    Image i = null;
    if (e == null || engine == null) return null;
    Object data = null;
    try {
      data = engine.eval("<graph-typeset>", -1, -1, "ExportString[ ToBoxes[" +
        e.toString() +", TraditionalForm], \"GIF\",  \"TransparentColor\"->GrayLevel[1]]");
      }
    catch (Exception ex) {}
    if (data != null && data instanceof String)
      return GUIKitUtils.createImage((String)data);
    return i;
    }
  
  protected void buildGraph(BasicGraphModel model, Object parent, Expr graphExpr) {
  	ArrayList nodes = new ArrayList();
		PropertyContainer root = (PropertyContainer)model.getRoot();
		root.setProperty("connectorType", "Straight");
		root.setProperty("directed", Boolean.FALSE);
  	if (graphExpr.length() > 2) {
  		for (int i = 3; i <= graphExpr.length(); ++i) {
  			Expr arg = graphExpr.part(i);
  			if (arg.head().equals(SYM_RULE)) {
  				String lhs = arg.part(1).toString();
  				Expr rhs = arg.part(2);
  				if (lhs.equals("EdgeDirection")) {
						root.setProperty("directed", rhs.equals(Expr.SYM_TRUE) ? Boolean.TRUE : Boolean.FALSE);
  				  }
  				else if (lhs.equals("EdgeColor")) {
  				  }
  			  }
  		  }
  	  }
		Rectangle2D newBounds = buildNodes(model, parent, nodes, graphExpr.part(2));
		buildEdges(model, parent, nodes, newBounds, graphExpr.part(1));
  	
  	// TODO make a GUI and call choice to let a user call Level, Random, or GridAnnealing
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {       
				GridAnnealingLayout l = new GridAnnealingLayout(new BasicLayoutTarget(getGraphController()));
				l.layout(getGraphModel().getRoot());
				}
			}); 
			
		// TODO figure out how to set viewport on view to match bounds of figures
		// since we are setting their initial coordinates from Mathematica coords
    // Look at trek code that seems to be doing this...
    }
  
  public Object createEdge(BasicGraphModel model, Object semantic, Expr ex, PropertyContainer root) {
		Object e = model.createEdge( semantic);
    if (ex == null) {
      Object defaultVal = root.getProperty("defaultValue");
      if (defaultVal != null) {
        ((PropertyContainer)e).setProperty("expr", defaultVal);
        if (defaultVal instanceof Expr)
          ((PropertyContainer)e).setProperty("image", createTypesetImage((Expr)defaultVal));
        }
      }
    else {
		  ((PropertyContainer)e).setProperty("expr", ex);
      ((PropertyContainer)e).setProperty("image", createTypesetImage(ex));
      }
		((PropertyContainer)e).setProperty("directed", root.getProperty("directed"));
		((PropertyContainer)e).setProperty("connectorType", root.getProperty("connectorType"));
		return e;
    }
  
	protected void buildArrayRules(BasicGraphModel model, Object parent, Expr arrRulesExpr) {
		HashMap nodes = new HashMap();
		PropertyContainer root = (PropertyContainer)model.getRoot();
		root.setProperty("connectorType", "Arc");
		root.setProperty("directed", Boolean.TRUE);
		
		for (int i = 1; i <= arrRulesExpr.length(); ++i) {
			Expr arg = arrRulesExpr.part(i);
			if (arg.head().equals(SYM_RULE)) {
				Expr lhs = arg.part(1);
				Expr rhs = arg.part(2);
				if (lhs.length() == 2 && lhs.head().equals(Expr.SYM_LIST)) {
					Expr nodeTail = lhs.part(1);
					Expr nodeHead = lhs.part(2);
					if (nodeTail.integerQ() && nodeHead.integerQ()) {
						try {
						  Integer tailIndex = new Integer(nodeTail.asInt());
						  Object tailNode = nodes.get(tailIndex);
							if (tailNode == null) {
								tailNode = model.createNode( tailIndex.toString());
								model.addNode(this, tailNode, model.getRoot());
								nodes.put(tailIndex, tailNode);
							  }
							Integer headIndex = new Integer(nodeHead.asInt());
							Object headNode = nodes.get(headIndex);
							if (headNode == null) {
								headNode = model.createNode( headIndex.toString());
								model.addNode(this, headNode, model.getRoot());
								nodes.put(headIndex, headNode);
							  }
							
							Object e = createEdge(model,  String.valueOf(i), rhs, root);
							((PropertyContainer)e).setProperty("directed", Boolean.TRUE);
							model.connectEdge(this, e, tailNode, headNode);
						  }
						catch (ExprFormatException ex) {}
					  }
					else {
						if (i == arrRulesExpr.length()) {
							root.setProperty("defaultValue", rhs);
						  }
					  }
				  }
			  }
		  }
  	
     //	TODO make a GUI and call choice to let a user call Level, Random, or GridAnnealing
		 SwingUtilities.invokeLater(new Runnable() {
			 public void run() {       
				 LevelLayout l = new LevelLayout(new BasicLayoutTarget(getGraphController()));
				 l.layout(getGraphModel().getRoot());
				 }
			 }); 
		}
		
	protected Rectangle2D buildNodes(BasicGraphModel model, Object parent, ArrayList nodes, Expr nodeListExpr) {
		Rectangle2D newBounds = null;
		
    int length = nodeListExpr.length();
    GraphController controller = getGraphController();
    for (int i = 1; i <= length; ++i) {
      Expr nodeExpr = nodeListExpr.part(i);
			Object node = null;
			if (nodeExpr.length() > 1) {
				// TODO process options
				}
			node = model.createNode( String.valueOf(i));
			nodes.add(node);
			
			Expr loc = nodeExpr.part(1);

			double newX = 0;
			double newY = 0;
			try {
				newX = loc.part(1).asDouble();
				newY = loc.part(2).asDouble();
			  } 
			catch (ExprFormatException ex){}
			
			controller.addNode(node, newX, newY);
			Figure f = controller.getFigure(node);
			if (newBounds == null) {
				Rectangle2D b = f.getBounds();
				newBounds = new Rectangle2D.Double(b.getX(), b.getY(), b.getWidth(), b.getHeight());
			  }
			Rectangle2D.union(newBounds, f.getBounds(), newBounds);
      }
  	return newBounds;
		}
  
	protected void buildEdges(BasicGraphModel model, Object parent, ArrayList nodes, 
      Rectangle2D newBounds, Expr edgeListExpr) {
		int length = edgeListExpr.length();
		GraphController controller = getGraphController();
		PropertyContainer root = (PropertyContainer)model.getRoot();
		Boolean defaultDirected = Boolean.TRUE;
		if (root.getProperty("directed") != null)
		  defaultDirected = (Boolean)root.getProperty("directed");
		for (int i = 1; i <= length; ++i) {
			Boolean directed = defaultDirected;
			Expr edgeExpr = edgeListExpr.part(i);
			Expr valExpr = null;
			if (edgeExpr.length() > 1) {
				for (int j = 2; j <= edgeExpr.length(); ++j) {
					Expr arg = edgeExpr.part(j);
					if (arg.head().equals(SYM_RULE)) {
						String lhs = arg.part(1).toString();
						Expr rhs = arg.part(2);
						if (lhs.equals("EdgeDirection")) {
							directed = rhs.equals(Expr.SYM_TRUE) ? Boolean.TRUE : Boolean.FALSE;
							}
						else if (lhs.equals("EdgeLabel")) {
							if (!rhs.equals(Expr.SYM_FALSE) && !rhs.equals(SYM_OFF))
								valExpr = rhs;
							}
						}
					}
				}
			Expr nodesExpr = edgeExpr.part(1);
			Object head = null;
			Object tail = null;
			try {
			  head = nodes.get(nodesExpr.part(2).asInt()-1);
			  tail = nodes.get(nodesExpr.part(1).asInt()-1);
			  }
			catch (ExprFormatException ex){}
			// TODO this needs to be expr or label obj
			Object e = model.createEdge( String.valueOf(i));
			
      if (valExpr != null) {
				((PropertyContainer)e).setProperty("expr", valExpr);
        ((PropertyContainer)e).setProperty("image", createTypesetImage(valExpr));
        }
			((PropertyContainer)e).setProperty("directed", directed);
			((PropertyContainer)e).setProperty("connectorType", root.getProperty("connectorType"));
			
			controller.addEdge(e, tail, head);

			Figure f = controller.getFigure(e);
			Rectangle2D.union(newBounds, f.getBounds(), newBounds);
			}
  	
		}
		
	public Expr getPart(int i) {
		Expr result = getExpr();
		if (result != null) return result.part(i);
		return null;
		}
	
	public Expr getPart(int[] ia) {
		Expr result = getExpr();
		if (result != null) return result.part(ia);
		return null;
		}
	
	public void setPart(int i, Expr e) {
		// TODO implement
			// if index is valid we can insert Node, Edge etc result within proper place in tree model
			// which could include negative indexing to append or fill?
		}
		
	public void setPart(int[] ia, Expr e) {
		// TODO implement
			// if index is valid we can insert Node, Edge etc result within proper place in tree model
			//		which could include negative indexing to append or fill?
		}
	
  }
