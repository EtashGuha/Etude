/*
 * @(#)TrekCanvas.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.diva;

import java.awt.event.ActionEvent;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.Iterator;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.ActionMap;
import javax.swing.InputMap;
import javax.swing.JComponent;
import javax.swing.KeyStroke;

import diva.canvas.Figure;
import diva.canvas.JCanvas;
import diva.canvas.interactor.SelectionModel;

/**
 * TrekCanvas.
 *
 * @version $Revision: 1.4 $
 */
public class TrekCanvas extends JCanvas {

  private static final long serialVersionUID = -1247987974456784948L;
    
  protected TrekPane trekPane;
	private boolean controlDown = false;
	private boolean shiftDown = false;
	private boolean spaceDown = false;
	
  public TrekCanvas() {
    super(new TrekPane(new TrekController()));
    trekPane = (TrekPane)getCanvasPane();
    
		CanvasKeyListener keyListener = new CanvasKeyListener();
    addKeyListener(keyListener);
    
		Action deleteTreksAction = new AbstractAction() {
            private static final long serialVersionUID = -1227987975451781948L;
			public void actionPerformed(ActionEvent e) {
				SelectionModel sm = trekPane.getTrekController().getSelectionModel();             
				ArrayList selKeys = new ArrayList();
				for(Iterator iter = sm.getSelection(); iter.hasNext();){
					 String key = trekPane.getTrekIdFromTarget((Figure)iter.next());
					 if (key != null) {
							selKeys.add(key);
							}
					 }
				//must clear selection first before removing figures from the layer
				sm.clearSelection();
				for(Iterator iter = selKeys.iterator(); iter.hasNext();) {
					trekPane.removeTrek((String)iter.next());
					}
				}
			}; 

		InputMap inputMap = getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
		ActionMap actionMap = getActionMap();
		if (inputMap != null && actionMap != null) {
			inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_DELETE,0), "DeleteTreks");
			inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_BACK_SPACE,0), "DeleteTreks");
			actionMap.put("DeleteTreks", deleteTreksAction);
			}
    
    }
  
  public void updateCursor() {
		trekPane.getTrekController().setToggleControls(controlDown, shiftDown);
		}
		
	public boolean isControlDown() {return controlDown;}
  public boolean isSpaceDown() {return spaceDown;}
  
  private class CanvasKeyListener extends KeyAdapter {
      public void keyPressed(KeyEvent e){
					if (e.isControlDown()) {
						if (!controlDown) {
							controlDown = true;
							trekPane.getTrekController().setToggleControls(controlDown, shiftDown);
							}
						}
				  if (e.isShiftDown()) {
						if (!shiftDown) {
							shiftDown = true;
							trekPane.getTrekController().setToggleControls(controlDown, shiftDown);
							}
				  	}
				  if (e.getKeyCode() == KeyEvent.VK_SPACE) {
				  	if (!spaceDown) spaceDown = true;
				  	}
     				super.keyPressed(e);
        }
			public void keyReleased(KeyEvent e){
				if (!e.isControlDown() && controlDown) {
					controlDown = false;
					trekPane.getTrekController().setToggleControls(controlDown, shiftDown);
					}
				if (!e.isShiftDown() && shiftDown) {
					shiftDown = false;
					trekPane.getTrekController().setToggleControls(controlDown, shiftDown);
					}
				if (spaceDown && (e.getKeyCode() == KeyEvent.VK_SPACE))
					spaceDown = false;
				super.keyReleased(e);
				}
    }
    
  public void reshape(int x, int y, int w, int h) {
    super.reshape(x, y, w, h);
    if (trekPane != null) 
      trekPane.canvasReshaped(x,y,w,h);
    }
}



