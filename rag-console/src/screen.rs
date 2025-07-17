use std::{
    fs::File,
    io::{stdin, stdout, Result, Write},
};

use termimad::{crossterm::style::{Attribute, Color}, rgb, terminal_size, FmtText, MadSkin};

const CLEAR_SCREEN_CMD: &str = "\x1B[3J\x1B[H\x1B[2J";

pub struct Screen {
    file: Option<File>,
    text_buffer: String,
    skin: MadSkin,
}

pub fn ai_terminal_skin() -> MadSkin {
    let mut skin = MadSkin::default();

    // Paragraphs: soft gray, left aligned
    skin.paragraph.align = termimad::Alignment::Left;
    skin.paragraph.set_fg(rgb(200, 200, 200)); // soft light gray

    // Bold text: used for emphasis in responses
    skin.bold.set_fg(rgb(0, 255, 180)); // teal green
    skin.bold.add_attr(Attribute::Bold);

    // Italics: used for soft notes, references
    skin.italic.set_fg(rgb(120, 220, 255)); // gentle cyan-blue

    // Inline code: highlight entities, API names, etc.
    skin.inline_code.set_fg(rgb(100, 255, 150)); // mint green
    skin.inline_code.set_bg(rgb(30, 30, 30));    // dark background
    skin.inline_code.add_attr(Attribute::Bold);

    // Code blocks: for retrieved context or AI-generated code
    skin.code_block.set_fg(rgb(180, 255, 180)); // bright green
    skin.code_block.set_bg(rgb(20, 20, 20));    // near-black background
    skin.code_block.add_attr(Attribute::Bold);

    // Headers (H1–H6): label sections like "Retrieved Context", "Answer", etc.
    skin.set_headers_fg(rgb(100, 220, 255)); // AI blue
    for header in skin.headers.iter_mut() {
        header.add_attr(Attribute::Bold);
    }

    // Bullet points: neutral tone for lists
    skin.bullet.set_fg(rgb(160, 160, 160));

    // Quotes: useful for highlighting retrieved content or citations
    skin.quote_mark.set_fg(rgb(100, 100, 100));
    // skin.quote_body.set_fg(rgb(180, 180, 180));
    // skin.quote_body.set_bg(rgb(35, 35, 35)); // muted dark
    
    // Optional: strike-through text support
    skin.strikeout.set_fg(Color::DarkGrey);
    skin.strikeout.add_attr(Attribute::CrossedOut);

    skin
}

impl Screen {
    pub fn new(file: Option<File>) -> Self {
        let text_buffer = String::from(CLEAR_SCREEN_CMD);
        let skin = ai_terminal_skin();
        Screen {
            file,
            text_buffer,
            skin,
        }
    }

    fn print_to_stdout(&mut self) {
        let (width, _) = terminal_size();
        let mut fmt_text =
            FmtText::from(&self.skin, &self.text_buffer, Some(width as usize)).to_string();
            fmt_text.pop();
        let _ = stdout().write(fmt_text.as_bytes());
        let _ = stdout().flush();
    }

    pub fn clear(&mut self) {
        self.text_buffer.clear();
        self.text_buffer.push_str(CLEAR_SCREEN_CMD);
        self.print_to_stdout();
    }

    pub fn read(&mut self, data: String) -> String {
        let mut usr_input = String::new();
        self.write_str(&data).expect("unable to write read data");
        stdin()
            .read_line(&mut usr_input)
            .expect("could not read user input");
        let cleaned_input = usr_input.trim();
        self.write_str(cleaned_input)
            .expect("unable to write cleaned input");
        self.write_str("\n\n").expect("unable to write newline");
        cleaned_input.to_owned()
    }

    pub fn read_human(&mut self) -> String {
        self.read("Human\t> ".into())
    }

    pub fn write_system(&mut self, message: &str) -> Result<usize> {
        self.write(format!("System\t> {message}").as_bytes())
    }

    pub fn write_str(&mut self, buf_str: &str) -> Result<usize> {
        self.write(buf_str.as_bytes())
    }
}

impl Write for Screen {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        for c in buf {
            self.text_buffer.push(char::from(c.clone()));
        }
        if let Some(mut file) = self.file.as_ref() {
            file.write(buf)?;
        }
        self.print_to_stdout();
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(mut file) = self.file.as_ref() {
            file.flush()?;
        }
        Ok(())
    }
}
