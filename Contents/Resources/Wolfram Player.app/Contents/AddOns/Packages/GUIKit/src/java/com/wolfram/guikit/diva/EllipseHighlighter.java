/*
 * @(#)EllipseHighlighter.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.diva;

import diva.canvas.DamageRegion;
import diva.canvas.Figure;
import diva.canvas.FigureDecorator;
import diva.canvas.toolbox.BasicEllipse;

import java.awt.Color;
import java.awt.Composite;
import java.awt.Paint;
import java.awt.Graphics2D;

import java.awt.geom.Rectangle2D;

/** A decorator figure that displays a highlight behind the
 * figure.
 *
 * @version	$Revision: 1.1 $
 */
public class EllipseHighlighter extends FigureDecorator {

    /* The compositing operation.
     */
    private Composite _composite = null;

    /* The "halo" size
     */
    private float _halo;

    /* The highlight paint
     */
    private Paint _paint;

    /** Create a new highlighter with a default paint and "halo"
     */
    public EllipseHighlighter () {
        this._paint = Color.yellow;
        this._halo = 3.0f;
    }

    /** Create a new highlighter with the given paint and "halo"
     */
    public EllipseHighlighter (Paint paint, float halo) {
        this._paint = paint;
        this._halo = halo;
    }

    /** Create a new highlighter with the given paint, "halo,"
     * and compositing operation.
     */
    public EllipseHighlighter (Paint paint, float halo, Composite composite) {
        this._paint = paint;
        this._halo = halo;
        this._composite = composite;
    }

    /** Get the composite.
     */
    public Composite getComposite () {
        return _composite;
    }

    /** Get the bounds. This is the child's bounding box stretched
     * by the "halo."
     */
    public Rectangle2D getBounds () {
        Rectangle2D b = getChild().getBounds();
        Rectangle2D bounds = new Rectangle2D.Double(
                b.getX() - _halo,
                b.getY() - _halo,
                b.getWidth() + 2 * _halo,
                b.getHeight() + 2 * _halo);

        return bounds;
    }

    /** Get the halo.
     */
    public float getHalo () {
        return _halo;
    }

    /** Get the paint.
     */
    public Paint getPaint () {
        return _paint;
    }

    /** Return false. This method always returns false, as it
     * is meaningless (and dangerous!) to be able to hit a highlight.
     */
    public boolean hit (Rectangle2D region) {
        return false;
    }

    /** Create a new instance of this highlighter. The new
     * instance will have the same paint, halo, and composite
     * as this one.
     */
    public FigureDecorator newInstance (Figure f) {
        return new EllipseHighlighter(_paint, _halo, _composite);
    }

    /** Paint the figure. This method first paints the highlight over
     * the contained figure's bounding box stretched by the halo. It
     * then paints the contained figure.
     */
    public void paint (Graphics2D g) {
        if (_composite != null) {
            g.setComposite(_composite);
        }
        //g.setPaint(_paint);

        // Draw the highlight
        // Rectangle2D bounds = getChild().getBounds();
        // FIXME: Can we draw non-rectangular highlights???
        Rectangle2D bounds = getChild().getShape().getBounds2D();
        double x = bounds.getX() - _halo;
        double y = bounds.getY() - _halo;
        double w = bounds.getWidth() + 2 * _halo;
        double h = bounds.getHeight() + 2 * _halo;

        BasicEllipse ep = new BasicEllipse(x,y,w,h, _paint);
        ep.paint(g);

        // Draw the child
        getChild().paint(g);
    }

    /** Receive repaint notification. This method generates another
     * repaint() call, with a larger region, in order to ensure
     * that the highlight is repainted.
     */
    public void repaint (DamageRegion d) {
        repaint();
    }

    /** Request a repaint of the figure and highlight.
     * This method reads the bounding box of the highlighted
     * figure, and requests a repaint of that box stretched
     * in each direction by the halo.
     */
    public void repaint () {
        Rectangle2D bounds = getChild().getBounds();
        double x = bounds.getX() - _halo;
        double y = bounds.getY() - _halo;
        double w = bounds.getWidth() + 2 * _halo;
        double h = bounds.getHeight() + 2 * _halo;

        getParent().repaint(DamageRegion.createDamageRegion(
                getTransformContext(),
                x,y,w,h));
    }
}



