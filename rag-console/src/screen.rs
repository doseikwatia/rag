use std::{
    fs::File,
    io::{stdin, stdout, Result, Write},
};

use termimad::{terminal_size, FmtText, MadSkin};

const CLEAR_SCREEN_CMD: &str = "\x1B[3J\x1B[H\x1B[2J";
pub struct Screen {
    file: Option<File>,
    text_buffer: String,
    skin: MadSkin,
}

impl Screen {
    pub fn new(file: Option<File>) -> Self {
        let text_buffer = String::from(CLEAR_SCREEN_CMD);
        let skin = MadSkin::default();
        Screen {
            file,
            text_buffer,
            skin,
        }
    }

    fn print_to_stdout(&mut self) {
        let (width, _) = terminal_size();
        let fmt_text = FmtText::from(&self.skin, &self.text_buffer, Some(width as usize));
        let _ = stdout().write(fmt_text.to_string().as_bytes());
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
        self.write_str("\n").expect("unable to write newline");
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
