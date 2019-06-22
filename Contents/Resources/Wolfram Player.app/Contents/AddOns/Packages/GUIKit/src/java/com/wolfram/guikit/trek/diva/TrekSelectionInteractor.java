/*
 * @(#)TrekSelectionInteractor.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.trek.diva;

import diva.canvas.event.LayerEvent;
import diva.canvas.interactor.SelectionInteractor;
import diva.canvas.interactor.SelectionModel;


/** 
 * TrekSelectionInteractor is attached to an object that can be put
 * into and out of a selection. Associated with each such role
 * is a selection model, which is the selection that figures
 * are added to or removed from.
 *
 * When a mouse pressed event has occured, all figures associated 
 * with the same SelectionModel will be unselected before the new
 * one is selected.  So, to make sure only one figure is selected 
 * at a time, do this:
 * SelectionModel m = new SelectionModel();
 * SelectionInteractor s1 = new SelectionInteractor(m);
 * SelectionInteractor s2 = new SelectionInteractor(m);
 *
 * <p> When an item is selected, events are then forwarded to
 * other interactors. If, however, the clicked-on figure has
 * just been removed from the figure, do not forward events.
 *
 * @version $Revision: 1.1 $
 */
public class TrekSelectionInteractor extends SelectionInteractor {

		private TrekPane trekPane;
		
		///////////////////////////////////////////////////////////////////
		////                         constructors                      ////

		/**
		 * Create a new SelectionInteractor with the given selection model
		 * and a null selection renderer.
		 */
	public TrekSelectionInteractor(TrekPane trekPane, SelectionModel model) {
		super(model);
		this.trekPane = trekPane;
		}
		
	public void mousePressed (LayerEvent event) {
		if (getSelectionFilter().accept(event) || getToggleFilter().accept(event))
			trekPane.getCanvas().setCursor(TrekController.SELECT_CURSOR);
			
		super.mousePressed(event);
		}
		
	public void mouseReleased (LayerEvent event) {
		trekPane.updateCursor();
		super.mouseReleased(event);
		}
	
}



