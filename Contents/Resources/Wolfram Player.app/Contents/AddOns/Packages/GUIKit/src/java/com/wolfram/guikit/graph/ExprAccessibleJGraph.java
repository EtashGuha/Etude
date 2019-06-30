/*
 * @(#)ExprAccessibleJGraph.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.graph;

import java.awt.Component;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.Iterator;

import com.oculustech.layout.OculusLayoutInfo;

import com.wolfram.guikit.type.ExprAccessible;
import com.wolfram.jlink.Expr;

import diva.canvas.Figure;
import diva.canvas.interactor.SelectionModel;
import diva.graph.GraphController;
import diva.graph.GraphModel;
import diva.graph.GraphPane;
import diva.graph.JGraph;

/**
 * ExprAccessibleJGraph extends JGraph
 */
public class ExprAccessibleJGraph extends JGraph implements ExprAccessible, OculusLayoutInfo {
  
    private static final long serialVersionUID = -1787977975476788978L;
    
	private static final ExprGraphTypeConvertor convertor = new ExprGraphTypeConvertor();
	
	public ExprAccessibleJGraph(GraphPane pane) {
		super(pane);
    addKeyListener(new DeleteSelectionListener());
		}
		
	public Expr getExpr() {
		if (getGraphPane() instanceof ExprAccessibleGraphPane) {
			return ((ExprAccessibleGraphPane)getGraphPane()).getExpr();
		  }
		else {
	 	  Object result = convertor.convert(getGraphPane().getGraphModel().getClass(), Expr.class, getGraphPane().getGraphModel());
			return (result instanceof Expr) ? (Expr)result : null;
		  }
		}
		
	public void setExpr(Expr e) {
		if (getGraphPane() instanceof ExprAccessibleGraphPane) {
		  ((ExprAccessibleGraphPane)getGraphPane()).setExpr(e);
		  }
		else {
		  Object result = convertor.convert(Expr.class, getGraphPane().getGraphModel().getClass(), e);
		  if (result != null && result instanceof GraphModel) {
			  setGraphPane(new GraphPane(getGraphPane().getGraphController(), (GraphModel)result));
		    }
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
	
  // methods of the OculusLayoutInfo interface to specify default stretching
  public int getXPreference() {return OculusLayoutInfo.CAN_BE_STRETCHED;}
  public int getYPreference() {return OculusLayoutInfo.CAN_BE_STRETCHED;}
  public Component getSameHeightAs() {return null;}
  public Component getSameWidthAs() {return null;}
  
  private class DeleteSelectionListener extends KeyAdapter {
      public void keyPressed(KeyEvent e){
          if(e.getKeyCode() == KeyEvent.VK_DELETE || e.getKeyCode() == KeyEvent.VK_BACK_SPACE){
              GraphController controller = getGraphPane().getGraphController();
              GraphModel model = getGraphPane().getGraphModel();
              SelectionModel sm = controller.getSelectionModel();             
              ArrayList selNodes = new ArrayList();
              for(Iterator iter = sm.getSelection(); iter.hasNext();){
                 Figure f = (Figure)iter.next();
                 if (f != null && f.getUserObject() != null) {
                   Object userObj = f.getUserObject();
                   if (model.isNode(userObj)) {
                     selNodes.add(userObj);
                     }
                   }
                 }

              for(Iterator iter = selNodes.iterator(); iter.hasNext();) {
                sm.removeSelection( controller.getFigure( iter.next()));
                }
                
              for(Iterator iter = selNodes.iterator(); iter.hasNext();) {
                if (controller instanceof DefaultGraphController)
                  ((DefaultGraphController)controller).removeNodeOnly(iter.next());
                else controller.removeNode(iter.next());
                }
                
              ArrayList selEdges = new ArrayList();
              for(Iterator iter = sm.getSelection(); iter.hasNext();){
                 Figure f = (Figure)iter.next();
                 if (f != null && f.getUserObject() != null) {
                   Object userObj = f.getUserObject();
                   if (model.isEdge(userObj))
                     selEdges.add(userObj);
                   }
                 }
                 
              //must clear selection first before removing figures from the layer
              sm.clearSelection();
              
              for(Iterator iter = selEdges.iterator(); iter.hasNext();) {
                controller.removeEdge(iter.next());
                }
            }
          else {
            super.keyPressed(e);
            }
        }
    }
    
  }
