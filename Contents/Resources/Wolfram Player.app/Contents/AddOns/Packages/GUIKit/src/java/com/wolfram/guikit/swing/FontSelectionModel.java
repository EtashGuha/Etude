/*
 * @(#)FontSelectionModel.java	1.10 03/01/23
 */
package com.wolfram.guikit.swing;

import javax.swing.event.ChangeListener;
import java.awt.Font;

/**
 * A model that supports selecting a <code>Font</code>.
 *
 * @version 1.0
 * @see java.awt.Font
 */
public interface FontSelectionModel {
    /**
     * Returns the selected <code>Font</code> which should be
     * non-<code>null</code>.
     *
     * @return  the selected <code>Font</code>
     * @see     #setSelectedFont
     */
    Font getSelectedFont();

    /**
     * Sets the selected font to <code>font</code>.
     * Note that setting the font to <code>null</code>
     * is undefined and may have unpredictable results.
     * This method fires a state changed event if it sets the
     * current font to a new non-<code>null</code> font.
     *
     * @param font the new <code>Font</code>
     * @see   #getSelectedFont
     * @see   #addChangeListener
     */
    void setSelectedFont(Font font);

    /**
     * Adds <code>listener</code> as a listener to changes in the model.
     * @param listener the <code>ChangeListener</code> to be added
     */
    void addChangeListener(ChangeListener listener);

    /**
     * Removes <code>listener</code> as a listener to changes in the model.
     * @param listener the <code>ChangeListener</code> to be removed
     */
    void removeChangeListener(ChangeListener listener);
}
