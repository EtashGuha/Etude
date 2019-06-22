package com.wolfram.databaselink.examples;

import java.awt.Color;
import java.awt.Component;

import javax.swing.ImageIcon;
import javax.swing.JTable;
import javax.swing.JLabel;
import javax.swing.border.EmptyBorder;
import javax.swing.table.TableCellRenderer;

class CommandRenderer implements TableCellRenderer
{
  protected JLabel label = null;
  protected Color selectionColor = new Color(226,189,242);

  public CommandRenderer()
  {
    label = new JLabel();
    label.setBorder(new EmptyBorder(15,15,15,15));
  }

  public Component getTableCellRendererComponent(JTable table,
                                                 Object value,
                                                 boolean isSelected,
                                                 boolean hasFocus,
                                                 int row,
                                                 int column)
  {
    Command command = (Command)value;
    ImageIcon icon = new ImageIcon(command.getImage());
    label.setIcon(icon);
    if(table.getRowHeight(row) != icon.getIconHeight() + 20)
      table.setRowHeight(row, icon.getIconHeight() + 20);
    label.setOpaque(true);
    if (isSelected) label.setBackground(selectionColor);
    return label;
  }
}
