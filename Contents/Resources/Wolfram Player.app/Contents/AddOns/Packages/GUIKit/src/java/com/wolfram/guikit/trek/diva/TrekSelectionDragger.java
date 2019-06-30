/*
 * @(#)TrekSelectionDragger.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.trek.diva;

import diva.canvas.Figure;
import diva.canvas.FigureDecorator;
import diva.canvas.FigureLayer;
import diva.canvas.GeometricSet;
import diva.canvas.OverlayLayer;

import diva.canvas.event.EventLayer;
import diva.canvas.event.LayerEvent;
import diva.canvas.event.MouseFilter;

import diva.canvas.interactor.Interactor;
import diva.canvas.interactor.DragInteractor;
import diva.canvas.interactor.SelectionInteractor;

import diva.util.CompoundIterator;
import java.awt.geom.Rectangle2D;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

/** 
 * TrekSelectionDragger is a class that implements rubber-banding on a canvas. It contains
 * references to one or more instances of SelectionInteractor, which it
 * notifies whenever dragging on the canvas covers or uncovers items.
 * The SelectionDragger requires three layers: an Event Layer,
 * which it listens to perform drag-selection, an OutlineLayer, on
 * which it draws the drag-selection box, and a FigureLayer, which it
 * selects figures on. It can also accept a GraphicsPane in its
 * constructor, in which case it will use the background event layer,
 * outline layer, and foreground event layer from that pane.
 *
 * @version $Revision: 1.1 $
 */
public class TrekSelectionDragger extends DragInteractor {

    /* The overlay layer
     */
    private OverlayLayer _overlayLayer;

    /* The event layer
     */
    private EventLayer _eventLayer;

    /* The figure layer
     */
    private FigureLayer _figureLayer;

    /* The set of valid selection interactors
     */
    private ArrayList _selectionInteractors = new ArrayList();

    /* The rubber-band
     */
    private Rectangle2D _rubberBand = null;

    /* The set of figures covered by the rubber-band
     */
    private GeometricSet _intersectedFigures;

    /** A hash-set containing those figures
     */
    private HashSet _currentFigures;

    /** A hash-set containing figures that overlap the rubber-band
     * but are not "hit"
     */
    private HashSet _holdovers;

    /* The origin points
     */
    private double _originX;
    private double _originY;
		private double prevX;
		private double prevY;
	
    /** The mouse filter for selecting items
     */
    private MouseFilter _selectionFilter = MouseFilter.selectionFilter;

    /** The mouse filter for toggling items
     */
    private MouseFilter _toggleFilter = MouseFilter.alternateSelectionFilter;

    /** The selection mode flags
     */
    private boolean _isSelecting;
    private boolean _isToggling;

		private TrekPane trekPane;
		
    ///////////////////////////////////////////////////////////////////
    ////                         constructors                      ////

    /**
     * Create a new SelectionDragger
     */
    public TrekSelectionDragger () {
        super();
    }

    /**
     * Create a new SelectionDragger attached to the given graphics
     * pane.
     */
    public TrekSelectionDragger (TrekPane trekPane) {
        super();
        this.trekPane = trekPane;
        setOverlayLayer(trekPane.getOverlayLayer());
        setEventLayer(trekPane.getBackgroundEventLayer());
        setFigureLayer(trekPane.getForegroundLayer());
    }

    ///////////////////////////////////////////////////////////////////
    //// public methods

    /**
     * Add a selection interactor to the list of valid interactor.
     * When drag-selecting, only figures that have an interactor
     * in this last are added to the selection model.
     */
    public void addSelectionInteractor (SelectionInteractor i) {
        if ( !(_selectionInteractors.contains(i))) {
            _selectionInteractors.add(i);
        }
    }

    /**
     * Clear the selection in all the relevant selection interactors.
     */
    public void clearSelection () {
        Iterator is = _selectionInteractors.iterator();
        while (is.hasNext()) {
            SelectionInteractor i = (SelectionInteractor) is.next();
            i.getSelectionModel().clearSelection();
        }
    }

    /**
     * Contract the selection by removing an item from it and
     * removing highlight rendering. If the figure is not in
     * the selection, do nothing.
     */
    public void contractSelection (SelectionInteractor i, Figure figure) {
        if (i.getSelectionModel().containsSelection(figure)) {
            i.getSelectionModel().removeSelection(figure);
        }
    }

    /**
     * Expand the selection by adding an item to it and adding
     * highlight rendering to it. If the
     * figure is already in the selection, do nothing.
     */
    public void expandSelection (SelectionInteractor i, Figure figure) {
        if ( !(i.getSelectionModel().containsSelection(figure))) {
            i.getSelectionModel().addSelection(figure);
        }
    }

    /**
     * Get the layer that drag rectangles are drawn on
     */
    public OverlayLayer getOverlayLayer () {
        return _overlayLayer;
    }

    /**
     * Get the layer that drag events are listened on
     */
    public EventLayer getEventLayer () {
        return _eventLayer;
    }

    /**
     * Get the layer that figures are selected on
     */
    public FigureLayer getFigureLayer () {
        return _figureLayer;
    }

    /**
     * Get the mouse filter that controls when this selection
     * filter is activated.
     */
    public MouseFilter getSelectionFilter () {
        return _selectionFilter;
    }

    /**
     * Get the mouse filter that controls the toggling of
     * selections
     */
    public MouseFilter getToggleFilter () {
        return _toggleFilter;
    }

    /** Reshape the rubber-band, swapping coordinates if necessary.
     * Any figures that are newly included or excluded from
     * the drag region are added to or removed from the appropriate
     * selection.
     */
    public void mouseDragged (LayerEvent event) {
        if (!isEnabled()) {
     			return;
        	}
        if (!_isToggling && !_isSelecting) {
	    		return;
        	}
				if (_rubberBand == null) {
	    		// This should never happen, but it does.
	    		return;
					}

				if (!trekPane.getCanvas().getCursor().equals(TrekController.SELECT_CURSOR))
					trekPane.getCanvas().setCursor(TrekController.SELECT_CURSOR);
			
        // Figure out the coordinates of the rubber band
        _overlayLayer.repaint(_rubberBand);
        
				// space held down translates
	 			// rubberband _originX, _originY by event-prev
	 			if (((TrekCanvas)trekPane.getCanvas()).isSpaceDown()) {
		 			_originX += (event.getLayerX() - prevX);
		 			_originY += (event.getLayerY() - prevY);
		 			}
		 
				prevX = event.getLayerX();
				prevY = event.getLayerY();
				 
				if (event.isAltDown())
					_rubberBand.setFrameFromCenter(_originX, _originY, event.getLayerX(), event.getLayerY());
				else
					_rubberBand.setFrameFromDiagonal(_originX, _originY, event.getLayerX(), event.getLayerY());

        _overlayLayer.repaint(_rubberBand);

        // Update the intersected figure set
        _intersectedFigures.setGeometry(_rubberBand);
				HashSet freshFigures = new HashSet();
				for (Iterator i = _intersectedFigures.figures(); i.hasNext(); ) {
	    		Figure f = (Figure)i.next();
	    		if (f instanceof FigureDecorator) {
						f = ((FigureDecorator)f).getDecoratedFigure();
	    			}
	   	 		if (f instanceof TrekFigure && ((TrekFigure)f).getSelectionFigure().hit(_rubberBand)) {
          	freshFigures.add(((TrekFigure)f).getSelectionFigure());
	    			} 
	    		else {
         		_holdovers.add(f);
	   				 }
					}
				for (Iterator i = ((HashSet)_holdovers.clone()).iterator(); i.hasNext(); ) {
	    		Figure f = (Figure)i.next();
	    		if (f instanceof TrekFigure && ((TrekFigure)f).getSelectionFigure().hit(_rubberBand)) {
						freshFigures.add(((TrekFigure)f).getSelectionFigure());
						_holdovers.remove(f);
	    			}
					}
        // stale = current-fresh;
				HashSet staleFigures = (HashSet) _currentFigures.clone();
				staleFigures.removeAll(freshFigures);
        // current = fresh-current
				HashSet temp = (HashSet) freshFigures.clone();
				freshFigures.removeAll(_currentFigures);
				_currentFigures = temp;

        // If in selection mode, add and remove figures
        if (_isSelecting) {
	    		// Add figures to the selection
	    		Iterator i = freshFigures.iterator();
	    		while (i.hasNext()) {
                Figure f = (Figure) i.next();
                Interactor r = f.getInteractor();
                if (r != null &&
                        r instanceof SelectionInteractor &&
                        _selectionInteractors.contains(r)) {
                    expandSelection((SelectionInteractor) r, f);
                }
	    			}

	    		// Remove figures from the selection
	    		i = staleFigures.iterator();
	    		while (i.hasNext()) {
                Figure f = (Figure) i.next();
                Interactor r = f.getInteractor();
                if (r != null &&
                        r instanceof SelectionInteractor &&
                        _selectionInteractors.contains(r)) {
                    contractSelection((SelectionInteractor) r, f);
                }
	    			}
        } else {
	    		// Toggle figures into and out of the selection
	    		Iterator i = new CompoundIterator(
                    freshFigures.iterator(),
                    staleFigures.iterator());
	    		while (i.hasNext()) {
                Figure f = (Figure) i.next();
                Interactor r = f.getInteractor();
                if (r != null &&
                        r instanceof SelectionInteractor &&
                        _selectionInteractors.contains(r)) {
                    SelectionInteractor s = (SelectionInteractor)r;
                    if (s.getSelectionModel().containsSelection(f)) {
                        contractSelection(s, f);
                    } else {
                        expandSelection(s, f);
                    }
                }
	    			}
        }
			// Consume the event
			if (isConsuming()) {
	    	event.consume();
				}
    }

    /** Clear the selection, and create the rubber-band
     */
    public void mousePressed (LayerEvent event) {
      if (!isEnabled()) {
        return;
      	}
      	
      // Check mouse event, set flags, etc
      _isSelecting = _selectionFilter.accept(event);
      _isToggling = _toggleFilter.accept(event);

      if (!_isToggling && !_isSelecting) {
				return;
      	}

			trekPane.getCanvas().setCursor(TrekController.SELECT_CURSOR);
		
      // Do it
      _originX = event.getLayerX();
      _originY = event.getLayerY();
			prevX = event.getLayerX();
			prevY = event.getLayerY();
					
      _rubberBand = new Rectangle2D.Double(_originX, _originY, 0.0, 0.0);

      _overlayLayer.add(_rubberBand);
      _overlayLayer.repaint(_rubberBand);

      _intersectedFigures =
			_figureLayer.getFigures().getIntersectedFigures(_rubberBand);
			_currentFigures = new HashSet();
			_holdovers = new HashSet();

      // Clear all selections
      if (_isSelecting) {
				clearSelection();
				}
				
		// Consume the event
		if (isConsuming()) {
    	event.consume();
			}
    }

    /** Delete the rubber-band
     */
    public void mouseReleased (LayerEvent event) {
			if (!isEnabled()) {
				return;
        }
			if (_rubberBand == null) {
	    	// This should never happen, but it does.
	    	return;
				}
			trekPane.updateCursor();
		
 			terminateDragSelection();

			// Consume the event
			if (isConsuming()) {
	    	event.consume();
				}
    }

    /**
     * Remove a selection interactor from the list of valid interactors.
     */
    public void removeSelectionInteractor (SelectionInteractor i) {
        if (_selectionInteractors.contains(i) ) {
            _selectionInteractors.remove(i);
        }
    }

    /**
     * Get the selection interactors
     */
    public Iterator selectionInteractors () {
        return _selectionInteractors.iterator();
    }

    /**
     * Set the layer that drag rectangles are drawn on
     */
    public void setOverlayLayer (OverlayLayer l) {
        _overlayLayer = l;
    }

    /**
     * Set the layer that drag events are listened on
     */
    public void setEventLayer (EventLayer l) {
        if (_eventLayer != null) {
            _eventLayer.removeLayerListener(this);
        }
        _eventLayer = l;
        _eventLayer.addLayerListener(this);
    }

    /**
     * Set the layer that figures are selected on
     */
    public void setFigureLayer (FigureLayer l) {
        _figureLayer = l;
    }

    /**
     * Set the mouse filter that controls when this selection
     * filter is activated.
     */
    public void setSelectionFilter(MouseFilter f) {
        _selectionFilter = f;
    }

    /**
     * Set the mouse filter that controls the toggling of
     * selections.
     */
    public void setToggleFilter(MouseFilter f) {
        _toggleFilter = f;
    }

    /** Terminate drag-selection operation. This must only be called
     * from events that are triggered during a drag operation.
     */
    public void terminateDragSelection () {
        if (!_isToggling && !_isSelecting) {
	    return;
        }

		_overlayLayer.repaint(_rubberBand);
		_overlayLayer.remove(_rubberBand);
		_rubberBand = null;
		_currentFigures = null;
		_holdovers = null;
    }
}



